//! The remembered fit choice (2026-11-25; `validate/references.md`, "Where a channel refit's time
//! goes" and "The remembered fit choice"): the trust recovery's degree-cap retry, kept with the
//! collection so that a launch tries it first instead of finding it again.
//!
//! # Why
//!
//! A launch refit is a fit from the data: the provider starts with `degree_cap` unset, so a
//! collection whose default-basis fit fails the training residuals pays that fit, then the
//! recovery's step 1 (`recover_trust`: the cap-6 retry, a cold fit) every time — measured on the
//! user's 5-element channel collection, 10.5 s of a 24.9 s launch refit. Within a session the cap
//! a recovery set already stands for the next refit (step 1b re-judges it when the capped fit
//! fails); across launches it was forgotten.
//!
//! # What is remembered
//!
//! `fit_choice.txt` in the collection's folder (`surrogate::collection`, "The remembered fit
//! choice"): the cap step 1 kept, and [`fit_choice_key`](SurrogateKs::fit_choice_key) — a
//! fingerprint of everything that decides the refit's first fit and its verdict, taken at the
//! refit's start (after the channel lifecycle, before any fit):
//! - every stored sample, bit for bit (atoms, positions, forces, energy on the fit's surface —
//!   the per-box rebase is in it —, weight, tag, force mask, energy flag), each one's quarantine
//!   code and whether it is box-affected: a sample appended, forgotten, restored, held, released,
//!   purged or rebased is another store state;
//! - the canonical and per-box references, the store's provenance (the force-noise prior), the
//!   policy (the residual and noise tiers judge the fit), the teacher tolerance;
//! - the default basis (species, n_max, l_max, the default degree cap, body 3/4, reach) and, on a
//!   channel collection, the mix the fit takes (K, W bit for bit, the represented flags): a new
//!   species, a W relearn, another K or another reach is another key;
//! - this build's version and the CPU's SIMD features (the dense kernels' results differ in the
//!   last bits between instruction sets).
//!
//! The store state counts as the same only when it is identical: the recovery's own rule is that
//! a fit that stands is the cold search of its store state (`recover_trust`, "Warm-started
//! trials"), so only an identical state is known to fail the default basis the same way. Any
//! change makes the key differ, and a stale entry is then inert (it never matches); a new one
//! replaces it when step 1 stands again.
//!
//! # How a refit uses it
//!
//! [`refit_fits`](SurrogateKs::refit_fits) computes the key when the refit's start is one the key
//! decides ([`fit_choice_entry`](SurrogateKs::fit_choice_entry)): a collection is attached; no
//! degree cap stands (a launch; in a session the standing cap is used as before); the first fit
//! is a rebuild searched from the data (no accumulator, or the species or references dirty — an
//! attach marks both; a warm incremental fit continues a search state the key leaves out); the basis has
//! the body-3 block below the recovery's cap (step 1's own condition); no fence vote is pending
//! (votes live in the session only, so a launch has none); no box-affected sample waits for its
//! exclusion (`exclude_box_samples` reads the last fit's body-4 degree, which the key leaves
//! out); the recovery is not switched off (tests). From such a start the default-basis fit, and
//! with it the release of held samples it reproduces, are functions of the key. Then:
//! - **remember**: a refit on the full path whose first fit failed, whose votes and release did
//!   nothing and whose recovery ended with step 1 standing records (key, cap 6) — the same start
//!   gives the same failing fit and the same empty release again;
//! - **reuse**: when the remembered key equals this refit's and no new W is staged, the refit sets
//!   the cap and fits once — the very fit step 1 makes (cap 6, a rebuild, searched from the data)
//!   — and keeps it only if it passes step 1's own check ([`recovered`](SurrogateKs::recovered):
//!   the training residuals, the noise and the trust rule's counts). It then stands with step 1's
//!   note and the refit goes on to the held-out guesses as after step 1. Otherwise the state on
//!   entry is restored (the fit, the codes, the dirty flags), the memory is dropped, and the full
//!   path runs from the default basis.
//!
//! So the fit that stands is the one the full recovery reaches, bit for bit (pinned:
//! `tests/fit_choice.rs`); only the default-basis fit before it is skipped. No trust check is
//! skipped or weakened: the default fit's failure is what the key remembers, and every check on
//! the fit that stands runs.
use super::*;

/// FNV-1a over bytes, 64-bit: a fingerprint stable across runs, builds and platforms (std's
/// `DefaultHasher` makes no such promise).
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }

    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }

    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
}

/// The SIMD features the dense kernels pick at run time (faer: `dev-reference.md`, "Dense linear
/// algebra on faer" — refits on CPUs of different instruction sets differ in the last bits, so a
/// verdict at the edge could too): a collection carried to another CPU is another key.
fn isa_tag() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let f = [
            ("sse4.2", std::arch::is_x86_feature_detected!("sse4.2")),
            ("avx", std::arch::is_x86_feature_detected!("avx")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
        ];
        f.iter().filter(|x| x.1).map(|x| x.0).collect::<Vec<_>>().join(",")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.to_string()
    }
}

impl SurrogateKs {
    /// The fingerprint of the refit's inputs (module doc, "What is remembered"); `None` without a
    /// collection or a store.
    pub(super) fn fit_choice_key(&self) -> Option<u64> {
        let sh = self.shared.as_ref()?;
        let store = self.store.as_ref()?;
        let mut h = Fnv::new();
        h.str("fit_choice key 1");
        h.str(env!("CARGO_PKG_VERSION"));
        h.str(&isa_tag());
        h.u64(store.samples.len() as u64);
        for (i, s) in store.samples.iter().enumerate() {
            h.u64(s.z.len() as u64);
            h.bytes(&s.z);
            for x in s.pos.iter().chain(&s.forces).flatten() {
                h.f64(*x);
            }
            h.f64(s.energy);
            h.f64(s.weight);
            h.str(&format!("{:?}", s.tag));
            match &s.force_mask {
                None => h.u64(0),
                Some(m) => {
                    h.u64(1 + m.len() as u64);
                    h.bytes(&m.iter().map(|&b| u8::from(b)).collect::<Vec<u8>>());
                }
            }
            // its code, and whether it is box-affected (the face density's one use in a fit)
            h.bytes(&[u8::from(s.use_energy), self.quarantine.get(i).copied().unwrap_or(Q_NONE), u8::from(self.is_box(i))]);
        }
        h.u64(self.refs.len() as u64);
        for (&z, &e) in &self.refs {
            h.bytes(&[z]);
            h.f64(e);
        }
        h.u64(sh.box_refs.len() as u64);
        for (&(z, n), &e) in &sh.box_refs {
            h.bytes(&[z]);
            h.u64(n as u64);
            h.f64(e);
        }
        h.str(&format!("{:?}", store.provenance));
        h.str(&format!("{:?}", self.policy));
        h.f64(self.teacher_tol);
        let species = self.store_species_sorted();
        let b = self.default_basis(&species);
        h.bytes(&b.species);
        for v in [b.n_max, b.l_max, b.degree_cap, usize::from(b.body3), usize::from(b.body4)] {
            h.u64(v as u64);
        }
        h.f64(b.r_cut);
        match self.collection_mix_for(&species) {
            None => h.u64(0),
            Some(m) => {
                h.u64(1 + m.k as u64);
                h.u64(m.w.len() as u64);
                m.w.iter().for_each(|&x| h.f64(x));
            }
        }
        // the mix the fit takes (a staged one first, as `collection_mix_for`): K, its rows, W and
        // the flags `mark_unrepresented` reads
        match sh.pending_mix.as_ref().map(|(m, _)| m).or(sh.mix.as_ref()) {
            None => h.u64(0),
            Some(m) => {
                h.u64(1 + m.k_requested as u64);
                h.u64(m.k as u64);
                h.u64(m.species.len() as u64);
                h.bytes(&m.species);
                h.u64(m.w.len() as u64);
                m.w.iter().for_each(|&x| h.f64(x));
                h.bytes(&m.represented.iter().chain(&m.n_w_met).map(|&b| u8::from(b)).collect::<Vec<u8>>());
            }
        }
        Some(h.0)
    }

    /// The store's species, sorted (the fit's species set).
    fn store_species_sorted(&self) -> Vec<u8> {
        let mut species: Vec<u8> = self.store.as_ref().map_or(Vec::new(), |s| s.samples.iter().flat_map(|x| x.z.iter().copied()).collect());
        species.sort_unstable();
        species.dedup();
        species
    }

    /// The key of this refit's start when its conditions hold (module doc, "How a refit uses
    /// it"), `None` otherwise: the remembered choice is then neither used nor recorded.
    pub(super) fn fit_choice_entry(&self) -> Option<u64> {
        if self.shared.is_none() || self.degree_cap.is_some() || self.recovery_skipped() || !self.vote_pending.is_empty() {
            return None;
        }
        // the first fit is a rebuild, searched from the data (a warm incremental fit's verdict
        // also depends on the search state it continues, which the key leaves out)
        if self.accumulator.is_some() && !self.species_dirty && !self.refs_dirty {
            return None;
        }
        let n = self.samples();
        if (0..n).any(|i| self.is_box(i) && self.quarantine.get(i).is_none_or(|&q| q == Q_NONE)) {
            return None;
        }
        let b = self.default_basis(&self.store_species_sorted());
        if !b.body3 || b.degree_cap >= RECOVERY_DEGREE_CAP {
            return None;
        }
        self.fit_choice_key()
    }

    /// The remembered cap tried first (module doc, "reuse"): `Some(report)` when it matches `key`
    /// and its fit passes step 1's check — the fit stands, with step 1's note; `None` when there is
    /// no matching memory (nothing changed) or its fit does not pass (the state on entry restored,
    /// the memory dropped: the caller runs the full path).
    pub(super) fn remembered_fit(&mut self, key: u64) -> Option<FitReport> {
        let sh = self.shared.as_ref()?;
        if sh.pending_mix.is_some() || sh.fit_choice != Some(surrogate::FitChoice { key, degree_cap: RECOVERY_DEGREE_CAP }) {
            return None;
        }
        let _prof = surrogate::prof::span("P2 remembered fit choice");
        let saved = (self.save_state(), self.quarantine.clone(), self.species_dirty, self.refs_dirty);
        // step 1's fit exactly: the cap, a rebuild, searched from the data
        self.degree_cap = Some(RECOVERY_DEGREE_CAP);
        self.species_dirty = true;
        match self.fit_store_deferred() {
            Ok(r) if self.recovered() => {
                self.recovery = Some(format!(
                    "{} — the choice remembered with the collection (the default basis failed them at an earlier refit of exactly these samples, so it was not fitted again)",
                    cap_retry_note()
                ));
                Some(r)
            }
            _ => {
                let (state, q, dirty, refs_dirty) = saved;
                self.restore_state(state);
                self.quarantine = q;
                self.species_dirty = dirty;
                self.refs_dirty = refs_dirty;
                self.set_fit_choice(None);
                None
            }
        }
    }

    /// After a full-path recovery from a start that met the conditions (`key`): records the cap
    /// when step 1 is what stood (module doc, "remember").
    pub(super) fn note_fit_choice(&mut self, key: u64) {
        if self.degree_cap == Some(RECOVERY_DEGREE_CAP) && self.recovery.as_deref() == Some(cap_retry_note().as_str()) {
            self.set_fit_choice(Some(surrogate::FitChoice { key, degree_cap: RECOVERY_DEGREE_CAP }));
        }
    }

    /// Sets the remembered choice; the collection's file follows at the next flush (a background
    /// refit's copy hands it to the install). A no-op when it is already the one held.
    fn set_fit_choice(&mut self, c: Option<surrogate::FitChoice>) {
        if let Some(sh) = self.shared.as_mut().filter(|sh| sh.fit_choice != c) {
            sh.fit_choice = c;
            sh.fit_choice_unsaved = true;
        }
    }

    /// The remembered choice (tests).
    #[cfg(test)]
    pub(super) fn fit_choice(&self) -> Option<surrogate::FitChoice> {
        self.shared.as_ref().and_then(|sh| sh.fit_choice)
    }
}
