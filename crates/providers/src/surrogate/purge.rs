//! **Purging forgotten samples** (2026-11-14; the Model tab's "Worst-fitting samples" › "Purge
//! forgotten (N)…"): the samples the user forgot ([`Q_FORGOTTEN`], `quarantine` module) are deleted
//! for good — from the store and, for the ones the attached shared collection holds, from its
//! `samples.bin` (`surrogate::Collection::purge`: rewritten atomically, the old file kept as
//! `samples.bin.bak`). Only forgotten samples: never a held, superseded, box-affected, draft, waiting
//! or lone one (the trust system's reversible states).
//!
//! # The fit stands, bit for bit
//!
//! A purge refits nothing. Forgotten samples are out of the fit once the refit after their Forget
//! has run (`forgotten_in_fit` = 0, which [`SurrogateKs::purge_refusal`] requires): the accumulator
//! never added them, the model, its posterior, the envelope, the null space and the grade's active
//! sets were built without them. So the model's predictions — energies, forces and their variances —
//! are bitwise those before the purge, and `model_gen` does not move (the leave-one-out cache, the
//! class verdicts and the views stay this fit's).
//!
//! # Ids are positional: everything is remapped
//!
//! A stored sample is its index in the store (`sim::SampleId::index`, with a check value of its
//! answer), and in collection mode the collection's own index (`shared::Link::In`). A purge removes
//! entries, so every later sample moves down; [`SurrogateKs::remap_after_purge`] takes the
//! old → new map and rewrites **every** index the provider holds: the store and its parallel
//! vectors (codes, lone codes, face densities, descriptors, the collection links with the
//! collection's own map, raw energies and boxes), the accumulated prefix and the fit's members,
//! the training residuals' and the misfit set's sample ids, the leave-one-out rows, the class rule's
//! worst samples and blockers, failed audits' kept indices, fence votes and the samples they kept,
//! lift votes, and the pass and teach-around starting counts; the readiness verdict is dropped (it
//! is recomputed at the next look) and the last pass's coverage recomputed. A
//! [`sim::SampleId`] taken before the purge carries the old index: its check value no longer
//! matches (or the index is past the end), so a stale Forget or Restore is refused (`check_id`).
//! A reference into a purged sample (a fence vote next to it) is dropped — a purged sample no
//! longer exists, so it fences nothing.
//!
//! # What a purge changes besides the store (the user's decision 2026-11-14: "purge lifts, warn")
//!
//! A purged sample no longer exists, so it no longer **fences**: geometries near it stop falling
//! back to DFT and the model answers there again (its fence votes, lift votes and kept-vote counts
//! go with it). It no longer counts against the trust recovery's 10 % cap on held + forgotten
//! samples either. The confirmation says so ("if they were bad data, keep them forgotten
//! instead"). A collection-mode checkpoint of an interrupted pass of this collection names counts
//! of the samples file as it was: the worker discards it after a purge.
//!
//! # When it is refused
//!
//! Nothing forgotten; a sampling pass running; a background refit running (its copy holds the old
//! indices); the refit after a Forget not run yet (the standing fit still holds a forgotten
//! sample: purging it would change what the fit was built from); and, with forgotten samples in
//! the collection, a collection handle without the lock — another copy of the app has it open — or
//! one turned read-only by a failed write. A refusal changes nothing; a failed collection write
//! changes nothing either (the collection rewrites first, the provider remaps only after it
//! succeeded).

use super::shared::Link;
use super::*;

/// The refusal while the fit still holds a forgotten sample.
pub const PURGE_WAIT_REFIT: &str = "the refit after the last Forget has not run yet (the fit still holds a forgotten sample) — purge once it has";

fn file_name(p: &std::path::Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |f| f.to_string_lossy().into_owned())
}

/// Keeps the entries of `v` whose index `keep` marks (an index past `keep` is kept).
fn retain_by<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        let k = keep.get(i).copied().unwrap_or(true);
        i += 1;
        k
    });
}

impl SurrogateKs {
    /// Why a purge would be refused now (module doc, "When it is refused"); `None`: it may run.
    pub fn purge_refusal(&self) -> Option<String> {
        if self.forgotten() == 0 {
            return Some("no forgotten samples".into());
        }
        if self.pass.is_some() {
            return Some("a sampling pass is running — purge once it is done".into());
        }
        if self.background.is_some() {
            return Some("a refit is running — purge once it has landed".into());
        }
        if self.forgotten_in_fit() > 0 {
            return Some(PURGE_WAIT_REFIT.into());
        }
        let sh = self.shared.as_ref()?;
        let in_collection = (0..self.quarantine.len()).any(|i| self.is_forgotten(i) && matches!(sh.link.get(i), Some(Link::In(_))));
        if in_collection {
            return sh.coll.purge_refusal();
        }
        None
    }

    /// **Purges** every forgotten sample (module doc): from the attached collection's file first —
    /// a failure there changes nothing —, then from the store, every index remapped. The fit is
    /// unchanged; nothing is refitted. `Err` with the reason, nothing changed.
    pub fn purge_forgotten(&mut self) -> Result<sim::PurgeReport, String> {
        if let Some(why) = self.purge_refusal() {
            return Err(why);
        }
        let n = self.samples();
        let gone: Vec<usize> = (0..n).filter(|&i| self.is_forgotten(i)).collect();
        // the collection's forgotten samples: each one's status written (a Forget whose write failed
        // earlier is written now, or the purge is refused)
        let mut in_coll: Vec<usize> = Vec::new();
        if let Some(sh) = self.shared.as_ref() {
            for &i in &gone {
                if let Some(&Link::In(k)) = sh.link.get(i) {
                    in_coll.push(k);
                }
            }
        }
        for &i in &gone {
            self.write_status_now(i).map_err(|e| format!("the forget of sample #{i} is not in the shared collection yet and cannot be written ({e}) — nothing purged"))?;
        }
        // the collection first: an `Err` there changed nothing; an `Ok` — with a warning or not —
        // is on disk and in the collection's memory, so the provider remaps whatever follows
        let (cmap, backup, warning) = match self.shared.as_mut() {
            Some(sh) if !in_coll.is_empty() => {
                let out = sh.coll.purge(&in_coll)?;
                (Some(out.map), out.backup, out.warning)
            }
            _ => (None, None, None),
        };
        let mut keep = vec![true; n];
        for &i in &gone {
            keep[i] = false;
        }
        self.remap_after_purge(&keep, cmap.as_deref());
        let purged = gone.len();
        let collection_mode = self.shared.is_some();
        let where_ = match (&backup, collection_mode) {
            (Some(b), _) if in_coll.len() == purged => format!(" from the shared collection's samples file (backup: {})", file_name(b)),
            (Some(b), _) => format!(" — {} from the shared collection's samples file (backup: {}), {} session-only", in_coll.len(), file_name(b), purged - in_coll.len()),
            (None, true) => " (session-only samples: the shared collection held none of them)".to_string(),
            (None, false) => " from the model (Save model writes the model file without them)".to_string(),
        };
        self.user_notes.push(format!(
            "surrogate: purged {purged} forgotten sample{}{where_}; the fit is unchanged, later samples are renumbered, and geometries near them no longer fall back to DFT",
            if purged == 1 { "" } else { "s" }
        ));
        if let Some(w) = &warning {
            self.user_notes.push(format!("surrogate: {w}"));
        }
        Ok(sim::PurgeReport { purged, from_collection: in_coll.len(), backup, collection_mode, warning })
    }

    /// How many of the forgotten samples the attached shared collection holds (the rest are
    /// session-only, or the model file's): the Model tab's confirmation words.
    pub fn forgotten_in_collection(&self) -> usize {
        let Some(sh) = self.shared.as_ref() else { return 0 };
        (0..self.quarantine.len()).filter(|&i| self.is_forgotten(i) && matches!(sh.link.get(i), Some(Link::In(_)))).count()
    }

    /// Rewrites every store index the provider holds after a purge (module doc, "Ids are
    /// positional"): `keep[i]` for stored sample `i`; `coll` the collection's own old → new map
    /// when its file was purged.
    pub(super) fn remap_after_purge(&mut self, keep: &[bool], coll: Option<&[Option<usize>]>) {
        let mut map: Vec<Option<usize>> = Vec::with_capacity(keep.len());
        let mut next = 0usize;
        for &k in keep {
            if k {
                map.push(Some(next));
                next += 1;
            } else {
                map.push(None);
            }
        }
        let m = |i: usize| map.get(i).copied().flatten();
        let below = |len: usize| keep[..len.min(keep.len())].iter().filter(|&&k| k).count();
        let id = |s: sim::SampleId| m(s.index).map(|index| sim::SampleId { index, key: s.key });
        if let Some(st) = self.store.as_mut() {
            retain_by(&mut st.samples, keep);
        }
        retain_by(&mut self.quarantine, keep);
        retain_by(&mut self.desc_cache, keep);
        retain_by(&mut self.face, keep);
        self.lone_codes = std::mem::take(&mut self.lone_codes).into_iter().filter_map(|(i, c)| m(i).map(|j| (j, c))).collect();
        // fences and votes: a purged sample fences nothing
        for v in &mut self.fence_votes {
            v.near = v.near.iter().filter_map(|&i| m(i)).collect();
        }
        self.fence_votes.retain(|v| !v.near.is_empty());
        for v in &mut self.lift_votes {
            v.near = v.near.iter().filter_map(|&i| m(i)).collect();
        }
        self.lift_votes.retain(|v| !v.near.is_empty());
        self.fence_kept = std::mem::take(&mut self.fence_kept).into_iter().filter_map(|(i, c)| m(i).map(|j| (j, c))).collect();
        self.vote_pending = std::mem::take(&mut self.vote_pending)
            .into_iter()
            .filter_map(|(i, near)| {
                let near: Vec<usize> = near.into_iter().filter_map(m).collect();
                Some((m(i)?, near)).filter(|(_, near)| !near.is_empty())
            })
            .collect();
        for f in self.failed_audits.iter_mut().chain(self.audit_fences.iter_mut()) {
            f.index = f.index.and_then(m);
        }
        // the fit's members and its readouts
        self.accumulated = below(self.accumulated);
        self.fit_members.0 = below(self.fit_members.0);
        retain_by(&mut self.fit_members.1, keep);
        for w in &mut self.class_worst {
            *w = w.and_then(|(i, r)| m(i).map(|j| (j, r)));
        }
        for b in &mut self.class_blockers {
            *b = b.iter().filter_map(|&(i, r)| m(i).map(|j| (j, r))).collect();
        }
        let remap_residuals = |t: &mut TrainingResiduals| {
            t.energy_at = t.energy_at.and_then(id);
            t.force_at = t.force_at.and_then(|(s, a)| id(s).map(|s| (s, a)));
        };
        if let Some(t) = self.training_residuals.as_mut() {
            remap_residuals(t);
        }
        if let Some(mf) = self.misfit.as_mut() {
            mf.samples = mf.samples.iter().filter_map(|&i| m(i)).collect();
            mf.atoms = mf.atoms.iter().filter_map(|&(i, a)| m(i).map(|j| (j, a))).collect();
            mf.worst = mf.worst.and_then(|(s, a)| id(s).map(|s| (s, a)));
            remap_residuals(&mut mf.outside);
        }
        if let Some(l) = self.loo_ratios.as_mut() {
            let rows_kept: Vec<bool> = l.rows.iter().map(|&(i, _)| m(i).is_some()).collect();
            l.rows = l.rows.iter().filter_map(|&(i, r)| m(i).map(|j| (j, r))).collect();
            retain_by(&mut l.tier, &rows_kept);
            retain_by(&mut l.tier_channel, &rows_kept);
        }
        // starting counts (store lengths)
        self.pass_start = self.pass_start.map(below);
        if let Some(t) = self.teach_start.as_mut() {
            t.samples_before = below(t.samples_before);
        }
        // the readiness verdict is recomputed (its refusal text may name a purged sample)
        self.strict_verdict = None;
        // what the last pass's samples became, over the renumbered store
        if self.pass_coverage.is_some() {
            self.pass_coverage = self.compute_pass_coverage();
        }
        if let Some(sh) = self.shared.as_mut() {
            retain_by(&mut sh.link, keep);
            retain_by(&mut sh.raw, keep);
            retain_by(&mut sh.box_n, keep);
            if let Some(cm) = coll {
                for l in &mut sh.link {
                    if let Link::In(k) = l {
                        *k = cm.get(*k).copied().flatten().expect("a kept sample's collection record is kept");
                    }
                }
                sh.pass_start = sh.pass_start.map(|p| cm[..p.min(cm.len())].iter().filter(|x| x.is_some()).count());
            }
        }
        // the views' cached curves name no sample, but their per-sample states are rebuilt
        self.curve_cache = None;
    }
}
