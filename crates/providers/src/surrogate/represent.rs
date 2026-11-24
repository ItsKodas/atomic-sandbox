//! Chemical channels: which species are represented, and the mix's lifecycle (plan
//! `docs/superpowers/plans/2026-11-22-chemical-channels.md` Tasks 5–6; spec
//! `docs/superpowers/specs/2026-11-22-chemical-channels-design.md`, "Physics review 2, 2026-11-23:
//! Task 5 replaced" items 1–10 with C2 and I5; review
//! `docs/superpowers/research/2026-11-23-chemical-channels-task4-task5-review.md` §C, R1–R8). Only a
//! **channel collection** (`shared` module doc, "The model basis") reaches anything here: with the
//! species basis every function is a no-op and the provider is bit for bit as before.
//!
//! # Refuse rather than widen
//!
//! The day-1 gate measured the frozen mix's first-order uncertainty far outside its validity
//! (η = 242–6 030), and W came out species-diagonal with **zero rows for Li and Cl** — they occur
//! only in diatomics, so their body-3 fibre carries no many-body information. A zero row is
//! harmless for a pair atom (body 2 carries its pair terms: HCl, HLi, LiO are accurate), and silent
//! for a many-body atom: O in HOCl sees only H in its channels, its descriptor sits inside the
//! training span, 5d and 5e cannot see the missing neighbour, and the fitted HOCl energy would be
//! absorbed into O{H}'s coefficients — water's. So the policy refuses: **no gᵀΣ_W g** enters an error
//! bar (Σ_W, η and ρ_t stay in `channels.txt` as diagnostics), and:
//!
//! 1. **The represented flag** (`surrogate::channels`, "Which species W represents"): decided when W
//!    is learnt, stored by Z in `channels.txt`, never read from W. A species is represented when
//!    (a) **N_W** — some many-body environment key (class 2) with it among the neighbours meets its
//!    sample floor n_k ≥ q_k in the low-order fit W is learnt from ([`Census`]: the existing per-key
//!    floor, `trust::key_column_counts`, in that fit's basis), and (b) its truncation residual
//!    ρ_t ≤ 0.01 (a NaN fails). A species without a row (a new element) is unrepresented.
//! 2. **W from the represented species only**: the learn passes N_W as the eligibility mask;
//!    unrepresented rows are exact zeros; W is the identity on the represented species when K covers
//!    them.
//! 3. **Rule 3b** (the decision list in the `surrogate` module doc; [`SurrogateKs::unrepresented_neighbour`]):
//!    a scene in which an atom with two or more neighbours within r_cut has an unrepresented species
//!    among them goes to the teacher, unconditionally (draft mode too, like rule 2: the model has no
//!    channel for that environment), and the answer is kept (thinned like rule 5's: a parked molecule
//!    must not add a sample per step) — those samples are how the species reaches N_W. One
//!    unrepresented neighbour of a pair atom is judged as before (body 2). The reason names the
//!    species and its N_W progress: "… (its best many-body environment has 3 of the 12 samples W
//!    needs)".
//! 4. **The channel fit leaves such samples out** ([`Q_UNREP`], set by
//!    [`mark_unrepresented`](SurrogateKs::mark_unrepresented) at every fit): a sample with **any**
//!    atom of two or more neighbours and an unrepresented one among them — any atom, not only the
//!    fitted ones of a capped segment: an edge atom's E_i enters its interior neighbours' forces. They
//!    still enter the low-order fit and count toward N_W. A relearn that represents the species takes
//!    them back.
//! 5. **Per-key floors (C2)** — `trust::key_column_counts_c2`: q_k = min(the channel columns k's
//!    atoms touch, the columns they touch in the shadow species basis — the species basis restricted
//!    to k's neighbours), counted over fitted samples only, so a key over at most K species is held
//!    to exactly today's floor and a dense W never raises a key to its centre's whole block. The
//!    same bound sets the whole-model floor (|∪ S_k|), `fit_floor`, the richer-basis floor and the
//!    sampling pass's floor (`sampling::min_samples_for_trust_channels`). An unseen key still needs
//!    the whole basis's Q′.
//! 6. **Novelty is unchanged**: 5d/5e cannot see a zero-row neighbour (rule 3b covers it) nor, under a
//!    dense W, a new combination of represented species (the unseen key's n_k = 0 floor covers it).
//!
//! # The lifecycle (item 8; review I5, R7)
//!
//! At the start of every refit ([`channel_lifecycle`](SurrogateKs::channel_lifecycle)) the provider
//! takes the **census** of the low-order fit's samples (codes `Q_NONE` and `Q_UNREP`: every sample a
//! fit could use, the channel fit's exclusions included) and learns W
//! (`surrogate::channels::learn_mix` with N_W as the mask, the collection's K and reach, the
//! surface's force-noise floor) when:
//! - **no W exists** and some species meets N_W (the first learn), or
//! - a species **first reaches N_W** (it had no row, or its row was learnt without N_W), or
//! - a species that met N_W at the last learn has **doubled its many-body sample count** m_t (the
//!   samples holding an atom of two or more neighbours with it among them; I5c names the element's
//!   informative count: a species that met N_W but failed ρ is included, so more data can represent
//!   it), or
//! - W has a row for a species the samples no longer hold (forgotten) and some species meets N_W —
//!   otherwise the fit is refused as before (`channel_refusal`).
//!
//! A new element **enters with an exact-zero row and causes no relearn** (`surrogate::collection::mix_for`)
//! until it reaches N_W. **Before any W** the fit's mix is one all-zero channel: every species
//! unrepresented, so the channel model is a body-2 model whose many-body atoms rule 3b hands to the
//! teacher — a diatomic collection answers from body 2, and its first many-body samples go towards
//! N_W. **A learnt W is staged** (`pending_mix`; physics review of Tasks 5–6, I1): the refit's fits
//! and the exclusion use it, while rule 3b and the evidence floor keep reading the committed W; it
//! is committed only once its first fit succeeds (`commit_pending_mix`), else dropped
//! (`drop_pending_mix`), and a background copy's W is installed only with a successful fit on it.
//! While the standing model's channels are not the committed W's, rule 3b refuses every many-body
//! atom. **A relearn that reproduces the same W** (species, K, W bits, represented flags; review I2)
//! is an ordinary refit: only its counts are written. A committed relearn writes `channels.txt` (a background refit's copy writes nothing: the install writes
//! it), rebuilds the fit, **clears the class hysteresis memory** (every key re-enters at ENTER) and
//! sets the mix's `history_from` to the next model generation, so the audit points measured under an
//! earlier W **leave the evidence counts, the windows and every λ** (`calibrate::calibrate_from`). A
//! learn that fails (a degenerate gap, a missing reference) is said and not retried until the census
//! changes. The low-order fit is the expensive part (`surrogate::channels` "Cost": ~60 s at S = 5 on
//! 1 000 six-atom samples); it runs only on a trigger, inside the refit (in the background with the
//! app's worker).
//!
//! **What K is.** The collection's key fixes the K asked for (`BasisKind::Channels { k }`); review
//! I3's choice of K by held-out RMSE is the gate's measurement, not the provider's.

use super::*;
use super::shared::SharedCollection;
use surrogate::channels::{learn_mix, LearntMix, LowOrder, MixSetup};
use surrogate::BasisKind;

/// The low-order fit's samples seen per species (module doc, "The lifecycle").
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Census {
    /// The species of the low-order fit's samples, sorted.
    pub(super) species: Vec<u8>,
    /// Per species: N_W met (a many-body key with it among the neighbours at n_k ≥ q_k).
    pub(super) n_w: Vec<bool>,
    /// Per species: m_t, the samples holding an atom of two or more neighbours with it among them.
    pub(super) many_body: Vec<usize>,
    /// Per species: the (n_k, q_k) of its many-body key nearest its floor (the largest n_k/q_k), for
    /// the words; `None` when it is no many-body atom's neighbour.
    pub(super) progress: Vec<Option<(usize, usize)>>,
}

impl Census {
    fn of(&self, z: u8) -> Option<usize> {
        self.species.iter().position(|&s| s == z)
    }
}

/// The trigger of a (re)learn ([`SurrogateKs::channel_lifecycle`]), in words; `None` when W stands.
pub(super) fn mix_due(census: &Census, mix: Option<&LearntMix>) -> Option<String> {
    let any = census.n_w.iter().any(|&b| b);
    let Some(m) = mix else {
        return any.then(|| format!("first learnt: {} met N_W", names(census.species.iter().zip(&census.n_w).filter(|(_, &b)| b).map(|(&z, _)| z))));
    };
    if !any {
        return None;
    }
    if let Some(&z) = m.species.iter().find(|z| census.of(**z).is_none()) {
        return Some(format!("relearnt: W had a row for {}, which the samples no longer hold", symbol(z)));
    }
    for (t, &z) in census.species.iter().enumerate() {
        let row = m.species.iter().position(|&s| s == z);
        let (was_nw, mb0) = row.map_or((false, 0), |r| (m.n_w_met[r], m.many_body[r]));
        if census.n_w[t] && !was_nw {
            return Some(format!("relearnt: {} first met N_W", symbol(z)));
        }
        if was_nw && census.many_body[t] >= 2 * mb0.max(1) {
            return Some(format!("relearnt: {}'s many-body samples doubled ({mb0} → {})", symbol(z), census.many_body[t]));
        }
    }
    None
}

fn names(z: impl Iterator<Item = u8>) -> String {
    z.map(symbol).collect::<Vec<_>>().join(", ")
}

/// Per atom with two or more neighbours within r_cut (r² < `rc2`): the first one with a neighbour
/// species `unrep` holds, as (atom, Z) — rule 3b's test and the channel fit's exclusion (module doc).
fn unrepresented_many_body(pos: &[[f64; 3]], z: &[u8], rc2: f64, unrep: impl Fn(u8) -> bool) -> Option<(usize, u8)> {
    if pos.len() != z.len() || pos.len() < 3 {
        return None;
    }
    let counts = neighbour_counts(pos, rc2);
    neighbour_species(pos, z, rc2).into_iter().enumerate().filter(|(i, _)| counts[*i] >= 2).find_map(|(i, (_, nb))| nb.into_iter().find(|&t| unrep(t)).map(|t| (i, t)))
}

impl SurrogateKs {
    /// The attached channel collection (`None`: a species collection, or none).
    pub(super) fn channel_collection(&self) -> Option<&SharedCollection> {
        self.shared.as_ref().filter(|sh| sh.coll.basis != BasisKind::Species)
    }

    /// The attached channel collection's K (its key's), for the sampling pass's C2 floor; `None`
    /// for the species basis.
    pub(super) fn channel_k(&self) -> Option<usize> {
        match self.channel_collection()?.coll.basis {
            BasisKind::Channels { k } => Some(k),
            BasisKind::Species => None,
        }
    }

    /// Whether `z` is represented by the attached channel collection's **committed** mix — the one
    /// the standing model was fitted with (`true` outside a channel collection: the species basis
    /// represents every species it has). Rule 3b reads this.
    pub(super) fn represents(&self, z: u8) -> bool {
        match self.channel_collection() {
            None => true,
            Some(sh) => sh.mix.as_ref().is_some_and(|m| m.represents(z)),
        }
    }

    /// Whether `z` is represented by the mix a fit now uses: a staged one (`pending_mix`) during
    /// the refit that learnt it, else the committed one. The exclusion (`mark_unrepresented`) reads
    /// this, so the staged fit leaves out exactly what its W cannot represent.
    fn represents_for_fit(&self, z: u8) -> bool {
        match self.channel_collection() {
            None => true,
            Some(sh) => sh.pending_mix.as_ref().map(|(m, _)| m).or(sh.mix.as_ref()).is_some_and(|m| m.represents(z)),
        }
    }

    /// The reach rule 3b, the exclusion and the census judge neighbours within (review M4: one
    /// source): the larger of the standing model's r_cut and the collection's, so the exclusion never
    /// sees fewer neighbours than the query rule.
    fn channel_r_cut(&self) -> f64 {
        self.model.as_ref().map_or(self.collection_r_cut, |m| m.basis.r_cut.max(self.collection_r_cut))
    }

    /// Rule 3b (module doc): the first atom with two or more neighbours and an unrepresented one
    /// among them, as (atom, that neighbour's Z); `None` outside a channel collection or where none.
    /// **Belt and braces** (review I1): while the standing model's channels are not the committed
    /// mix's (a W given or learnt whose fit has not stood yet), every species counts as
    /// unrepresented — every many-body atom is refused until a fit on the committed W stands.
    pub(super) fn unrepresented_neighbour(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<(usize, u8)> {
        self.channel_collection()?;
        let r = self.channel_r_cut();
        let stale = self.model.as_ref().is_some_and(|m| m.basis.channels() != self.committed_mix_for(&m.basis.species).as_ref());
        unrepresented_many_body(pos, z, r * r, |t| stale || !self.represents(t))
    }

    /// Rule 3b's refusal words: the atom, the species and its N_W progress.
    pub(super) fn unrepresented_words(&self, atom: usize, zu: u8) -> String {
        let el = symbol(zu);
        let progress = self.channel_collection().and_then(|sh| sh.census.as_ref()).and_then(|c| c.of(zu).map(|t| (c.n_w[t], c.progress[t])));
        let why = match (progress, self.channel_collection().and_then(|sh| sh.mix.as_ref()).and_then(|m| m.species.iter().position(|&s| s == zu).map(|r| (m.n_w_met[r], m.truncation_residual[r])))) {
            (_, Some((true, rho))) => format!("W was learnt with {el} but its truncation residual ρ {rho:.2} is above {} — it is relearnt once {el}'s many-body samples double", surrogate::channels::RHO_MAX),
            (Some((_, Some((n, q)))), _) => format!("its best many-body environment has {n} of the {q} samples W needs"),
            _ => format!("no many-body sample has {el} as a neighbour yet"),
        };
        format!(
            "surrogate: atom {atom} has two or more neighbours and one is {el}, which the chemical channels do not represent yet ({why}) — the teacher answers and its answer is kept for the next learn of W"
        )
    }

    /// Whether stored sample `s` has an atom of two or more neighbours with an unrepresented
    /// species among them (module doc, item 4).
    fn unrepresentable(&self, s: &Sample, rc2: f64) -> bool {
        unrepresented_many_body(&s.pos, &s.z, rc2, |t| !self.represents_for_fit(t)).is_some()
    }

    /// Item 4 of the module doc, at every fit: in a channel collection, every `Q_NONE` sample that
    /// is unrepresentable becomes [`Q_UNREP`], and every `Q_UNREP` one that no longer is (a relearn
    /// represented its species) goes back to the code a release gives. A change below the
    /// accumulated prefix makes the fit a rebuild (`species_dirty`). `true` when one did. A no-op
    /// outside a channel collection.
    pub(super) fn mark_unrepresented(&mut self) -> bool {
        if self.channel_collection().is_none() {
            return false;
        }
        let Some(store) = self.store.as_ref() else { return false };
        let _prof = surrogate::prof::span("P3 mark_unrepresented");
        let r = self.channel_r_cut();
        let rc2 = r * r;
        let mut moves: Vec<(usize, u8)> = Vec::new();
        for (i, s) in store.samples.iter().enumerate() {
            match self.quarantine.get(i).copied().unwrap_or(Q_NONE) {
                Q_NONE if self.unrepresentable(s, rc2) => moves.push((i, Q_UNREP)),
                Q_UNREP if !self.unrepresentable(s, rc2) => moves.push((i, self.released_code(i))),
                _ => {}
            }
        }
        let mut below = false;
        for (i, q) in moves {
            if self.quarantine.len() <= i {
                self.quarantine.resize(i + 1, Q_NONE);
            }
            self.quarantine[i] = q;
            below |= i < self.accumulated;
        }
        if below {
            self.species_dirty = true;
        }
        below
    }

    /// The first audit-history generation of the standing W (module doc, "The lifecycle"): 0 outside
    /// a channel collection or before W is learnt.
    pub(super) fn history_floor(&self) -> u64 {
        self.channel_collection().and_then(|sh| sh.mix.as_ref()).map_or(0, |m| m.history_from)
    }

    /// The stored samples of the low-order fit W is learnt from: those a fit could use (`Q_NONE`)
    /// and those the channel fit leaves out for an unrepresented neighbour (`Q_UNREP`).
    fn low_order_members(&self) -> Vec<usize> {
        let n = self.store.as_ref().map_or(0, |s| s.len());
        (0..n).filter(|&i| matches!(self.quarantine.get(i).copied().unwrap_or(Q_NONE), Q_NONE | Q_UNREP)).collect()
    }

    /// The census of the low-order fit's samples (module doc); `None` outside a channel collection
    /// or without such samples.
    pub(super) fn channel_census(&self) -> Option<Census> {
        self.channel_collection()?;
        let _prof = surrogate::prof::span("P1 channel_census (low-order key counts)");
        let store = self.store.as_ref()?;
        let members = self.low_order_members();
        let mut species: Vec<u8> = members.iter().flat_map(|&i| store.samples[i].z.iter().copied()).collect();
        species.sort_unstable();
        species.dedup();
        if species.is_empty() {
            return None;
        }
        let r_cut = self.channel_r_cut();
        let low = LowOrder::REVIEW_I4.basis(&species, r_cut);
        let counts = trust::key_column_counts(&low, members.iter().map(|&i| &store.samples[i]));
        let s_n = species.len();
        let (mut n_w, mut progress) = (vec![false; s_n], vec![None::<(usize, usize)>; s_n]);
        for (k, &(n, q)) in counts.per_key.iter().filter(|(k, _)| k.class == 2) {
            for t in &k.neighbours {
                let Some(ti) = species.iter().position(|s| s == t) else { continue };
                n_w[ti] |= n >= q;
                let better = progress[ti].is_none_or(|(n0, q0)| (n as f64) * (q0.max(1) as f64) > (n0 as f64) * (q.max(1) as f64));
                if better {
                    progress[ti] = Some((n, q));
                }
            }
        }
        let rc2 = r_cut * r_cut;
        let mut many_body = vec![0usize; s_n];
        for &i in &members {
            let s = &store.samples[i];
            let nc = neighbour_counts(&s.pos, rc2);
            let mut seen = vec![false; s_n];
            for (j, (_, nb)) in neighbour_species(&s.pos, &s.z, rc2).into_iter().enumerate() {
                if nc[j] < 2 {
                    continue;
                }
                for t in nb {
                    if let Some(ti) = species.iter().position(|&x| x == t) {
                        seen[ti] = true;
                    }
                }
            }
            for (m, s) in many_body.iter_mut().zip(seen) {
                *m += usize::from(s);
            }
        }
        Some(Census { species, n_w, many_body, progress })
    }

    /// The lifecycle step at the start of a refit (module doc, "The lifecycle"): the census, and a
    /// learn of W when one is due. A no-op outside a channel collection.
    pub(super) fn channel_lifecycle(&mut self) {
        #[cfg(test)]
        if self.mix_learn_off {
            return;
        }
        let Some(census) = self.channel_census() else { return };
        let sh = self.shared.as_mut().expect("a channel collection");
        sh.census = Some(census.clone());
        let Some(why) = mix_due(&census, sh.mix.as_ref()) else { return };
        // a learn that failed is tried again only when its inputs changed in substance (review M6):
        // the N_W set, or a many-body count doubled since — not at every new sample
        if sh.learn_failed.as_ref().is_some_and(|f| !retry_due(f, &census)) {
            return;
        }
        let BasisKind::Channels { k } = sh.coll.basis else { return };
        let ref_energy: Option<Vec<f64>> = census.species.iter().map(|z| self.refs.get(z).copied().filter(|e| e.is_finite())).collect();
        let Some(ref_energy) = ref_energy else {
            self.user_notes.push(format!("chemical channels: W is due ({why}) but a species' isolated-atom reference is missing — learnt once it arrives"));
            return;
        };
        let store = self.store.as_ref().expect("a census has samples");
        let members = self.low_order_members();
        let samples: Vec<&Sample> = members.iter().map(|&i| &store.samples[i]).collect();
        let ionic = store.provenance.ionic.clone();
        let mut setup = MixSetup::new(&census.species, &ref_energy, self.collection_r_cut);
        setup.fit = FitOptions { sigma_f_min: force_noise_prior(&ionic), ..FitOptions::default() };
        setup.eligible = Some(census.n_w.clone());
        let _prof = surrogate::prof::span("P1 channel_lifecycle (learn W)");
        // the refit picture: the low-order fit's gathering and search are the mix's (display only)
        if let Some(p) = &self.meter {
            p.refit_mix(true);
        }
        let learnt = learn_mix(&samples, &setup, k);
        if let Some(p) = &self.meter {
            p.refit_mix(false);
        }
        match learnt {
            Ok(mut mix) => {
                mix.many_body = census.many_body.clone();
                let sh = self.shared.as_mut().expect("a channel collection");
                if let Some(old) = sh.mix.as_ref().filter(|old| same_basis(old, &mix)) {
                    // the same W on the same represented species (review I2): an ordinary refit —
                    // only the counts the triggers read move; the evidence and the memory stand
                    let mut kept = old.clone();
                    kept.many_body = mix.many_body;
                    kept.n_w_met = mix.n_w_met;
                    let mut note = format!("chemical channels: W {why}, and came out unchanged — its counts updated, the trust evidence stands");
                    if !self.refit_fork {
                        if let Err(e) = sh.coll.save_mix(&kept) {
                            note.push_str(&format!(" (not written to the collection: {e})"));
                        }
                    }
                    sh.mix = Some(kept);
                    sh.learn_failed = None;
                    self.user_notes.push(note);
                } else {
                    mix.history_from = self.model_gen + 1;
                    sh.pending_mix = Some((mix, why));
                    self.species_dirty = true;
                }
            }
            Err(e) => {
                self.user_notes.push(format!("chemical channels: W could not be learnt ({why}): {e} — the standing mix is kept, and the learn is tried again when the samples change"));
                self.shared.as_mut().expect("a channel collection").learn_failed = Some(census);
            }
        }
    }

    /// Commits the mix staged by [`channel_lifecycle`](Self::channel_lifecycle) once its first fit
    /// has succeeded (review I1; module doc, "The lifecycle"): written to `channels.txt` (not by a
    /// background refit's copy: its install writes it, and only with the fit), rule 3b and the
    /// evidence floor now read it, the class memory cleared (the trust passes of the fit that stands
    /// run after this). A failed write keeps the mix for this session and says so. A no-op without a
    /// staged mix.
    pub(super) fn commit_pending_mix(&mut self) {
        let fork = self.refit_fork;
        let Some(sh) = self.shared.as_mut() else { return };
        let Some((mix, why)) = sh.pending_mix.take() else { return };
        let why = &why;
        let mut note = format!("chemical channels: W {why} (K = {}): {}", mix.k, represented_words(&mix));
        if !fork {
            if let Err(e) = sh.coll.save_mix(&mix) {
                note.push_str(&format!(" — not written to the collection ({e}): used for this session only"));
            }
        }
        sh.mix = Some(mix);
        sh.learn_failed = None;
        note.push_str(" — every environment's trust verdict starts over, and the checks measured under the earlier W leave the evidence");
        // (the staged fit was a rebuild already; a later one re-marks with the committed flags)
        self.user_notes.push(note);
        self.species_dirty = true;
        self.class_memory.clear();
        self.fit_floor_cache.lock().unwrap().clear();
        self.last_variance = None;
    }

    /// Drops a staged mix whose first fit failed (review I1): the committed W stands — rule 3b, the
    /// evidence and `channels.txt` as before — and the learn is not retried until its inputs change
    /// (`learn_failed`). The codes the staged fit marked are marked again at the next fit.
    pub(super) fn drop_pending_mix(&mut self, e: &FitError) {
        let Some(sh) = self.shared.as_mut() else { return };
        let Some((_, why)) = sh.pending_mix.take() else { return };
        sh.learn_failed = sh.census.clone();
        self.species_dirty = true;
        self.user_notes.push(format!("chemical channels: W {why}, but the fit on it failed ({e}) — the standing W is kept, and the learn is tried again when the samples change in substance"));
    }
}

/// Whether two mixes give the same channel basis on the same represented species (review I2): the
/// species, K, W bit for bit and the represented flags.
fn same_basis(a: &LearntMix, b: &LearntMix) -> bool {
    let bits = |m: &LearntMix| m.w.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    a.species == b.species && a.k == b.k && a.represented == b.represented && bits(a) == bits(b)
}

/// Whether a failed learn's census `failed` should be retried at `now` (review M6): another species
/// set or N_W set, or a many-body count doubled since.
fn retry_due(failed: &Census, now: &Census) -> bool {
    failed.species != now.species || failed.n_w != now.n_w || failed.many_body.iter().zip(&now.many_body).any(|(&f, &n)| n >= 2 * f.max(1))
}

/// "represented H, O; not represented: Cl (N_W unmet), S (ρ 0.43 > 0.01)".
pub(super) fn represented_words(mix: &LearntMix) -> String {
    let rep: Vec<&str> = mix.species.iter().zip(&mix.represented).filter(|(_, &r)| r).map(|(&z, _)| symbol(z)).collect();
    let not: Vec<String> = (0..mix.species.len())
        .filter(|&t| !mix.represented[t])
        .map(|t| {
            let el = symbol(mix.species[t]);
            if mix.n_w_met[t] {
                format!("{el} (ρ {:.2} > {})", mix.truncation_residual[t], surrogate::channels::RHO_MAX)
            } else {
                format!("{el} (N_W unmet)")
            }
        })
        .collect();
    let mut w = format!("represented {}", if rep.is_empty() { "none".to_string() } else { rep.join(", ") });
    if !not.is_empty() {
        w.push_str(&format!("; not represented: {}", not.join(", ")));
    }
    w
}
