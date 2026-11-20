//! The collection migration's words (Fermi–Dirac spec `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`
//! §7.8; plan 2026-11-13 Task 16): the Model tab's offer, its folded line after **Later**, the progress line (the
//! Model tab's collection section and the Activity tab's pass card) and the stopped line of a pass that will go on.
//! Pure functions of `sim::MigrationStatus` (`Trainable::migration`), so the copy is unit tested; the drawing is
//! `dock::model`'s and `activity::jobs`'s. The summary is the provider's (`surrogate::migration::summary_text`, the text
//! the collection's `README.txt` carries too).
//!
//! The copy is the spec's, verbatim where §7.8 gives it. Numbers are grouped as the spec writes them ("2 412",
//! "1 579 K"), with a no-break space (U+00A0) so a label never wraps mid-number; the electronic temperature is
//! `ui::pseudo::t_el_kelvin` (kT = `qm::ELECTRON_KT`, plan ruling R19). A time is "1 h 30 min" ([`hm`]).

use sim::{MigrationPhase, MigrationStatus};

/// The offer's first sentence (drawn strong).
pub const OFFER_TITLE: &str = "This collection was taught with the previous solver.";
/// The offer's buttons, in order: [`Action::StartMigration`](super::Action) and [`Action::MigrationLater`](super::Action).
pub const OFFER_BUTTONS: [&str; 2] = ["Re-solve now", "Later"];
/// The progress line's head (spec §7.8) and the Activity tab's pass-card title.
pub const RESOLVING: &str = "Re-solving for the new solver";

/// "2 412" for 2412, with a no-break space (U+00A0; review M8, re-review R1): `surrogate::migration`'s, which the summary uses.
pub use surrogate::migration::spaced;

/// A GPU time as the offer and the progress line word it: "1 h 30 min" (from an hour on, to 5 min: it is an
/// estimate), "52 min", "under a minute".
pub fn hm(s: f64) -> String {
    let s = if s.is_finite() { s.max(0.0) } else { 0.0 };
    if s < 60.0 {
        return "under a minute".into();
    }
    let min = (s / 60.0).round() as u64;
    if min < 60 {
        return format!("{min} min");
    }
    let min = ((s / 300.0).round() as u64) * 5;
    match (min / 60, min % 60) {
        (h, 0) => format!("{h} h"),
        (h, m) => format!("{h} h {m} min"),
    }
}

/// `hm(s)` after an approximation word — "About 1 h 30 min", "≈ 52 min" — but never before "under a minute", which is
/// approximate already: "Under a minute" / "under a minute" (the word's case kept; Task 16 re-review R3).
pub fn approx(s: f64, word: &str) -> String {
    match hm(s) {
        w if w == "under a minute" && word.starts_with(char::is_uppercase) => "Under a minute".into(),
        w if w == "under a minute" => w,
        w => format!("{word} {w}"),
    }
}

/// "H, N, O".
fn species(z: &[u8]) -> String {
    z.iter().map(|&z| units::elements::symbol(z)).collect::<Vec<_>>().join(", ")
}

/// Whether the Model tab offers the migration: the attached collection has a predecessor with samples and no pass
/// has started (`MigrationPhase::Offered`).
pub fn offer_shown(m: Option<&MigrationStatus>) -> bool {
    m.is_some_and(|m| m.state == MigrationPhase::Offered && m.samples > 0)
}

/// The offer's text after its title (spec §7.8).
pub fn offer_body(m: &MigrationStatus) -> String {
    let what = if m.species.is_empty() { format!("{} samples", spaced(m.samples as u64)) } else { format!("{} · {} samples", species(&m.species), spaced(m.samples as u64)) };
    let time = m.eta_s.map_or(String::new(), |s| format!(" {} of GPU time.", approx(s, "About")));
    format!(
        "Atomic Sandbox now fills electron levels with Fermi–Dirac smearing at {} K, which fixes SCF stalls on loose clusters. Your collection \"{what}\" (models/collections/{}) was taught with the old rule. Its samples need one re-solve with the new solver before the model uses them. Most will come back identical within solver noise. Open-shell and nearly degenerate ones change by up to a few mHa; those are checked against the rest before they rejoin the fit.{time} You can pause it and keep working. The original folder is kept unchanged, and older versions keep using it.",
        spaced(u64::from(super::pseudo::t_el_kelvin())),
        m.source
    )
}

/// The whole offer: [`OFFER_TITLE`] and [`offer_body`] (the card draws them apart: the title strong).
#[cfg(test)]
pub fn offer_text(m: &MigrationStatus) -> String {
    format!("{OFFER_TITLE} {}", offer_body(m))
}

/// The offer folded by **Later**: one line, beside its Re-solve now.
pub fn later_line(m: &MigrationStatus) -> String {
    let time = m.eta_s.map_or(String::new(), |s| format!(" ({} of GPU time)", approx(s, "about")));
    format!("This collection was taught with the previous solver: {} samples need one re-solve with the new solver{time}.", spaced(m.samples as u64))
}

/// "380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated".
fn class_counts(m: &MigrationStatus) -> String {
    let [u, t, s, n] = m.counts.map(|c| spaced(c as u64));
    format!("{u} identical · {t} changed only by −TS · {s} changed state · {n} not migrated")
}

/// "≈ 52 min left", "under a minute left" ([`approx`]).
fn left(s: f64) -> String {
    format!("{} left", approx(s, "≈"))
}

/// The progress line's counts after its head, without the time left: "sample 412 of 2 412 · 380 identical · … · 0 not
/// migrated" (the Activity tab's and the 3D view's pass cards show them; their time left is the card's own).
pub fn progress_counts(m: &MigrationStatus) -> String {
    format!("sample {} of {} · {}", spaced(m.done as u64), spaced(m.samples as u64), class_counts(m))
}

/// The progress line (spec §7.8, the Activity tab): "Re-solving for the new solver: sample 412 of 2 412 · 380
/// identical · … · ≈ 52 min left".
pub fn progress_line(m: &MigrationStatus) -> String {
    let eta = m.eta_s.map_or(String::new(), |s| format!(" · {}", left(s)));
    format!("{RESOLVING}: {}{eta}", progress_counts(m))
}

/// A pass started and stopped (Cancel, or the app closed): where it goes on — "sample k of n", k the next sample to
/// re-solve, as the provider's note numbers it (review M3) — and that it goes on.
pub fn stopped_line(m: &MigrationStatus) -> String {
    let left = m.eta_s.map_or(String::new(), |s| format!(" ({} of GPU time left)", approx(s, "about")));
    format!(
        "{RESOLVING} stopped at sample {} of {} ({}). It goes on from there at the next start of the app, or now with Resume{left}.",
        spaced(m.done as u64 + 1),
        spaced(m.samples as u64),
        class_counts(m)
    )
}

/// The summary once the pass is complete (spec §7.8): the provider's `surrogate::migration::summary_text`, the text the
/// collection's `README.txt` carries too.
pub fn summary_text(m: &MigrationStatus) -> Option<&str> {
    m.summary.as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use surrogate::{MigrationLog, MigrationState, NotMigrated};

    /// "2 412" with the no-break space (review M8, re-review R1).
    fn g(s: &str) -> String {
        s.replace("_", "\u{a0}")
    }

    /// The fixture predecessor of spec §7.8: "H, N, O · 2 412 samples", 1 h 30 min a priori.
    fn offered() -> MigrationStatus {
        MigrationStatus {
            source: "lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef".into(),
            samples: 2412,
            done: 0,
            counts: [0; 4],
            eta_s: Some(5400.0),
            state: MigrationPhase::Offered,
            species: vec![1, 7, 8],
            summary: None,
        }
    }

    /// The offer is spec §7.8's, verbatim: the species and the count from the predecessor, its folder, the estimate as
    /// "About 1 h 30 min", T_el 1 579 K, and the two buttons.
    #[test]
    fn migration_offer_text() {
        let m = offered();
        assert_eq!(
            offer_text(&m),
            g("This collection was taught with the previous solver. Atomic Sandbox now fills electron levels with Fermi–Dirac smearing at 1_579 K, which fixes SCF stalls on loose clusters. Your collection \"H, N, O · 2_412 samples\" (models/collections/lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef) was taught with the old rule. Its samples need one re-solve with the new solver before the model uses them. Most will come back identical within solver noise. Open-shell and nearly degenerate ones change by up to a few mHa; those are checked against the rest before they rejoin the fit. About 1 h 30 min of GPU time. You can pause it and keep working. The original folder is kept unchanged, and older versions keep using it.")
        );
        assert_eq!(OFFER_TITLE, "This collection was taught with the previous solver.");
        assert_eq!(OFFER_BUTTONS, ["Re-solve now", "Later"]);
        assert!(offer_shown(Some(&m)));
        // the estimate's words
        assert_eq!(hm(5400.0), "1 h 30 min");
        assert_eq!(hm(3120.0), "52 min");
        assert_eq!(hm(7260.0), "2 h");
        assert_eq!(hm(5530.0), "1 h 30 min", "an hour and more: to 5 min");
        assert_eq!(hm(20.0), "under a minute");
        // never "about under a minute" (re-review R3): the offer, Later and the stopped line for a small collection
        let small = MigrationStatus { samples: 12, eta_s: Some(20.0), ..m.clone() };
        assert!(offer_body(&small).contains(" Under a minute of GPU time. "), "{}", offer_body(&small));
        assert!(later_line(&small).ends_with("(under a minute of GPU time)."), "{}", later_line(&small));
        let stopped = MigrationStatus { state: MigrationPhase::Paused, ..small.clone() };
        assert!(stopped_line(&stopped).ends_with("(under a minute of GPU time left)."), "{}", stopped_line(&stopped));
        for t in [offer_body(&small), later_line(&small), stopped_line(&stopped), progress_line(&MigrationStatus { state: MigrationPhase::Running, ..small })] {
            assert!(!t.contains("about under") && !t.contains("About under") && !t.contains("≈ under"), "{t}");
        }
        assert_eq!((approx(5400.0, "About"), approx(20.0, "About"), approx(20.0, "≈")), ("About 1 h 30 min".into(), "Under a minute".into(), "under a minute".into()));
        assert_eq!(spaced(2412), "2\u{a0}412", "a no-break space: never wrapped mid-number");
        // epaint 0.36's row-break candidates are `c.is_whitespace() && c != '\u{a0}'` (text_layout.rs): a grouped
        // number holds none, so a wrapping label never breaks it (re-review R1: U+202F is whitespace, so it would)
        for n in [2412u64, 1_234_567, 1579] {
            assert!(!spaced(n).chars().any(|c| c.is_whitespace() && c != '\u{a0}'), "{:?}", spaced(n));
        }
        assert!('\u{202f}'.is_whitespace(), "why not U+202F: epaint wraps at it");
        assert_eq!(spaced(412), "412");
        assert_eq!(spaced(1_234_567), g("1_234_567"));
        // Later folds it into one line, the count and the estimate kept
        assert_eq!(later_line(&m), g("This collection was taught with the previous solver: 2_412 samples need one re-solve with the new solver (about 1 h 30 min of GPU time)."));
    }

    /// The Activity line of spec §7.8, verbatim; no time left once there is none to say, and no "≈" before "under a
    /// minute" (review M9); the counts alone carry no time (review M1); the stopped line numbers the next sample as the
    /// provider's note does (review M3).
    #[test]
    fn migration_progress_line() {
        let m = MigrationStatus { done: 412, counts: [380, 22, 10, 0], eta_s: Some(52.0 * 60.0), state: MigrationPhase::Running, ..offered() };
        assert_eq!(progress_line(&m), g("Re-solving for the new solver: sample 412 of 2_412 · 380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated · ≈ 52 min left"));
        assert_eq!(progress_line(&MigrationStatus { eta_s: None, ..m.clone() }), g("Re-solving for the new solver: sample 412 of 2_412 · 380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated"));
        assert!(progress_line(&MigrationStatus { eta_s: Some(20.0), ..m.clone() }).ends_with("· under a minute left"));
        assert_eq!(progress_counts(&m), g("sample 412 of 2_412 · 380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated"));
        let stopped = MigrationStatus { state: MigrationPhase::Paused, eta_s: Some(52.0 * 60.0), ..m };
        assert_eq!(stopped_line(&stopped), g("Re-solving for the new solver stopped at sample 413 of 2_412 (380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated). It goes on from there at the next start of the app, or now with Resume (about 52 min of GPU time left)."));
    }

    /// The log of spec §7.8's summary.
    fn reviewed() -> MigrationLog {
        MigrationLog {
            source_slug: "lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef".into(),
            source_key: String::new(),
            build: "0.2.0".into(),
            date: "2026-11-20".into(),
            source_len: 0,
            source_hash: 0,
            cursor: 2412,
            state: MigrationState::Reviewed,
            counts: [2302, 64, 41, 5],
            not_migrated: vec![
                (3, NotMigrated::Stalled, "the cold solve failed: SCF stalled".into()),
                (9, NotMigrated::BoxAffected, "box-affected".into()),
                (17, NotMigrated::Stalled, "the cold solve failed: SCF stalled".into()),
                (40, NotMigrated::BoxAffected, "box-affected".into()),
                (77, NotMigrated::BoxAffected, "box-affected".into()),
            ],
            references: vec![(1, 57, Some(-0.4791), Some(-0.4791)), (8, 57, Some(-15.8), Some(-15.8095))],
            class_u_max: (0.0, 0.0),
            review: (0..41).map(|k| (3000 + k, if k < 35 { 0.5 } else { 3.0 }, k < 35)).collect(),
            waiting: Vec::new(),
            pending: None,
            orphaned: Vec::new(),
        }
    }

    /// The summary of spec §7.8 (`surrogate::migration::summary_text`, the Model tab's and `README.txt`'s one text):
    /// "identical within solver noise", "kept out", the old slug and the date; the review's waiting samples (stray
    /// atoms; Task 15) on their own clause; the not-migrated reasons by their codes (fix round 1, (c)); the references.
    #[test]
    fn migration_summary_text() {
        let m = reviewed();
        let want = g("Migrated from lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef on 2026-11-20:\n\
- 2_302 identical within solver noise (kept as they were);\n\
- 64 changed only by the electronic-entropy term −TS (re-labelled, kept in the fit);\n\
- 41 changed state (re-solved and reviewed: 35 rejoined the fit because the model without each predicts it within its bound; 6 are kept out, and the model falls back to DFT near them);\n\
- 5 not migrated (2 stalled, 3 now box-affected; they remain in the old collection).\n\
\n\
References re-solved (the new F against the old E): H +0.0 mHa, O −9.5 mHa. The audit history starts empty, so trust is rebuilt from new audits.");
        let s = surrogate::summary_text(&m).expect("complete");
        assert_eq!(s, want);
        // the Model tab shows the provider's text as it is
        let status = MigrationStatus { state: MigrationPhase::Reviewed, summary: Some(s.clone()), ..offered() };
        assert_eq!(summary_text(&status), Some(want.as_str()));
        for words in ["identical within solver noise", "kept out", "lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef"] {
            assert!(s.contains(words), "{words:?} in\n{s}");
        }
        assert!(!s.contains("discarded"), "never 'discarded'");
        assert_eq!(surrogate::summary_text(&MigrationLog { state: MigrationState::Running, ..m.clone() }), None, "no summary while the pass runs");
        // the review's waiting samples (stray atoms, Task 15 review M1) and the ones it never judges get their clauses
        let mut waiting = m.clone();
        waiting.state = MigrationState::Complete;
        waiting.review.truncate(39);
        waiting.waiting = vec![5000, 5001];
        let w = surrogate::summary_text(&waiting).unwrap();
        assert!(w.contains("- 41 changed state (re-solved and reviewed: 35 rejoined the fit because the model without each predicts it within its bound; 4 are kept out, and the model falls back to DFT near them; 2 waiting (stray atoms));"), "{w}");
        let mut unjudged = m.clone();
        unjudged.review.truncate(38);
        let u = surrogate::summary_text(&unjudged).unwrap();
        assert!(u.contains("3 are kept out, and the model falls back to DFT near them; 3 not reviewed (they kept an earlier Forget or supersede, or wait for an isolated-atom reference));"), "{u}");
        // before the first review: what the review will do
        let before = MigrationLog { state: MigrationState::Complete, review: Vec::new(), ..m.clone() };
        assert!(surrogate::summary_text(&before).unwrap().contains("- 41 changed state (re-solved; the next refit reviews them: those the model without each predicts within its bound rejoin the fit, the others are kept out and the model falls back to DFT near them);"));
        // one of each: singular words; a reference that did not solve
        let one = MigrationLog {
            counts: [1, 0, 1, 1],
            review: vec![(3000, 3.0, false)],
            not_migrated: vec![(0, NotMigrated::Multiplicity, "solved at multiplicity 3, recorded 1".into())],
            references: vec![(8, 57, Some(-15.8), None)],
            ..m.clone()
        };
        let s = surrogate::summary_text(&one).unwrap();
        assert!(s.contains("- 1 changed state (re-solved and reviewed: 0 rejoined the fit because the model without each predicts it within its bound; 1 is kept out, and the model falls back to DFT near it);"), "{s}");
        assert!(s.contains("- 1 not migrated (1 solved at another multiplicity; it remains in the old collection)."), "{s}");
        assert!(s.contains("References re-solved (the new F against the old E): O not solved."), "{s}");
        // nothing changed state, nothing not migrated, no references (review M5)
        let none = MigrationLog { counts: [2, 0, 0, 0], review: Vec::new(), not_migrated: Vec::new(), references: Vec::new(), ..m };
        let s = surrogate::summary_text(&none).unwrap();
        assert!(s.contains("- 0 changed state;\n- 0 not migrated.") && s.contains("References re-solved: none."), "{s}");
    }

    /// No offer without samples to migrate (a predecessor with 0 samples), nor once a pass has started or completed;
    /// no status at all is no offer.
    #[test]
    fn no_offer_without_samples_to_migrate() {
        assert!(!offer_shown(None));
        assert!(!offer_shown(Some(&MigrationStatus { samples: 0, ..offered() })), "a predecessor with 0 samples");
        for state in [MigrationPhase::Running, MigrationPhase::Paused, MigrationPhase::Complete, MigrationPhase::Reviewed] {
            assert!(!offer_shown(Some(&MigrationStatus { state, ..offered() })), "{state:?}");
        }
    }
}
