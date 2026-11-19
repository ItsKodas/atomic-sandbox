//! The collection migration's words (Fermi–Dirac spec `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`
//! §7.8; plan 2026-11-13 Task 16): the Model tab's offer, its folded line after **Later**, the progress line (the
//! Model tab's collection section and the Activity tab's pass card), the stopped line of a pass that will go on, and
//! the summary. Pure functions of `sim::MigrationStatus` (`Trainable::migration`), so the copy is unit tested; the
//! drawing is `dock::model`'s and `activity::jobs`'s.
//!
//! The copy is the spec's, verbatim where §7.8 gives it. Numbers are grouped with a space as the spec writes them
//! ("2 412", "1 579 K"); the electronic temperature is `ui::pseudo::t_el_kelvin` (kT = `qm::ELECTRON_KT`, plan ruling
//! R19). A time is "1 h 30 min" ([`hm`]).

use sim::{MigrationPhase, MigrationStatus};

/// The offer's first sentence (drawn strong).
pub const OFFER_TITLE: &str = "This collection was taught with the previous solver.";
/// The offer's buttons, in order: [`Action::StartMigration`](super::Action) and [`Action::MigrationLater`](super::Action).
pub const OFFER_BUTTONS: [&str; 2] = ["Re-solve now", "Later"];
/// The progress line's head (spec §7.8) and the Activity tab's pass-card title.
pub const RESOLVING: &str = "Re-solving for the new solver";

/// "2 412" for 2412 (spec §7.8 groups thousands with a space).
pub fn spaced(n: u64) -> String {
    let d = n.to_string();
    let mut out = String::with_capacity(d.len() + d.len() / 3);
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

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
    let time = m.eta_s.map_or(String::new(), |s| format!(" About {} of GPU time.", hm(s)));
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
    let time = m.eta_s.map_or(String::new(), |s| format!(" (about {} of GPU time)", hm(s)));
    format!("This collection was taught with the previous solver: {} samples need one re-solve with the new solver{time}.", spaced(m.samples as u64))
}

/// "380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated".
fn class_counts(m: &MigrationStatus) -> String {
    let [u, t, s, n] = m.counts.map(|c| spaced(c as u64));
    format!("{u} identical · {t} changed only by −TS · {s} changed state · {n} not migrated")
}

/// "sample 412 of 2 412".
fn sample_of(m: &MigrationStatus) -> String {
    format!("sample {} of {}", spaced(m.done as u64), spaced(m.samples as u64))
}

/// The progress line's counts after its head: "sample 412 of 2 412 · 380 identical · … · ≈ 52 min left" (the Activity
/// tab's pass card shows them under its title).
pub fn progress_counts(m: &MigrationStatus) -> String {
    let left = m.eta_s.map_or(String::new(), |s| format!(" · ≈ {} left", hm(s)));
    format!("{} · {}{left}", sample_of(m), class_counts(m))
}

/// The progress line (spec §7.8, the Activity tab): "Re-solving for the new solver: sample 412 of 2 412 · 380
/// identical · …".
pub fn progress_line(m: &MigrationStatus) -> String {
    format!("{RESOLVING}: {}", progress_counts(m))
}

/// A pass started and stopped (Cancel, or the app closed): where it stands, and that it goes on.
pub fn stopped_line(m: &MigrationStatus) -> String {
    let left = m.eta_s.map_or(String::new(), |s| format!(" (about {} of GPU time left)", hm(s)));
    format!("{RESOLVING} stopped at {} ({}). It goes on from there at the next start of the app, or now with Resume{left}.", sample_of(m), class_counts(m))
}

/// A not-migrated sample's reason (the pass's words, `providers::sampling::migrate`) in a few words.
fn reason_words(why: &str) -> &'static str {
    if why.contains("SCF stalled") {
        "stalled"
    } else if why.contains("SCF did not converge") {
        "not converged"
    } else if why.starts_with("box-affected") {
        "now box-affected"
    } else if why.starts_with("box status unknown") {
        "box status unknown"
    } else if why.starts_with("the solve warned") {
        "with a solver warning"
    } else if why.starts_with("the solve is not trained on") {
        "not trainable"
    } else if why.starts_with("solved at") {
        "solved at another multiplicity"
    } else if why == surrogate::collection::REFUSE_LONE {
        "lone atoms"
    } else if why.contains("forced the multiplicity") {
        "with a forced multiplicity"
    } else if why.starts_with("its status code") {
        "with a status no collection holds"
    } else if why.starts_with("its scene is not on") {
        "on another surface"
    } else if why.starts_with("the cold solve failed") {
        "failed to solve"
    } else {
        "for another reason"
    }
}

/// "2 stalled, 3 now box-affected", in the order the reasons first appear.
fn reasons(whys: &[String]) -> String {
    let mut tally: Vec<(&'static str, usize)> = Vec::new();
    for w in whys.iter().map(|w| reason_words(w)) {
        match tally.iter_mut().find(|(k, _)| *k == w) {
            Some(e) => e.1 += 1,
            None => tally.push((w, 1)),
        }
    }
    tally.iter().map(|(w, n)| format!("{} {w}", spaced(*n as u64))).collect::<Vec<_>>().join(", ")
}

/// "+0.0", "−9.5": a signed value with spec §7.8's minus sign.
fn signed(x: f64, digits: usize) -> String {
    let t = format!("{:.*}", digits, x.abs());
    if x < 0.0 && t.chars().any(|c| c.is_ascii_digit() && c != '0') {
        format!("−{t}")
    } else {
        format!("+{t}")
    }
}

/// The summary once the pass is complete (spec §7.8; the Model tab).
pub fn summary_text(m: &MigrationStatus) -> String {
    let [u, t, s, n] = m.counts;
    let [released, kept, waiting] = m.review;
    let sp = |k: usize| spaced(k as u64);
    let date = if m.date.is_empty() { String::new() } else { format!(" on {}", m.date) };
    let mut out = format!("Migrated from {}{date}:\n", m.source);
    out += &format!("- {} identical within solver noise (kept as they were);\n", sp(u));
    out += &format!("- {} changed only by the electronic-entropy term −TS (re-labelled, kept in the fit);\n", sp(t));
    let changed = if m.state == MigrationPhase::Complete && m.review == [0; 3] {
        "re-solved; the next refit reviews them: those the model without each predicts within its bound rejoin the fit, the others are kept out and the model falls back to DFT near them".to_string()
    } else {
        let kept_words = match kept {
            0 => "none is kept out".to_string(),
            1 => "1 is kept out, and the model falls back to DFT near it".to_string(),
            k => format!("{} are kept out, and the model falls back to DFT near them", sp(k)),
        };
        let mut w = format!("re-solved and reviewed: {} rejoined the fit because the model without each predicts it within its bound; {kept_words}", sp(released));
        if waiting > 0 {
            w += &format!("; {} waiting (stray atoms)", sp(waiting));
        }
        let unjudged = s.saturating_sub(released + kept + waiting);
        if unjudged > 0 {
            w += &format!("; {} not reviewed (they kept an earlier Forget or supersede, or wait for an isolated-atom reference)", sp(unjudged));
        }
        w
    };
    out += &format!("- {} changed state ({changed});\n", sp(s));
    out += &if n == 0 {
        "- 0 not migrated.\n".to_string()
    } else {
        let remain = if n == 1 { "it remains" } else { "they remain" };
        format!("- {} not migrated ({}; {remain} in the old collection).\n", sp(n), reasons(&m.not_migrated))
    };
    out.push('\n');
    out += &if m.references.is_empty() {
        "References re-solved: none.".to_string()
    } else {
        let mut per_z: std::collections::BTreeMap<u8, usize> = Default::default();
        for r in &m.references {
            *per_z.entry(r.0).or_default() += 1;
        }
        let words: Vec<String> = m
            .references
            .iter()
            .map(|&(z, box_n, old, new)| {
                let name = if per_z[&z] > 1 { format!("{} ({box_n}³)", units::elements::symbol(z)) } else { units::elements::symbol(z).to_string() };
                match (old, new) {
                    (Some(o), Some(x)) => format!("{name} {} mHa", signed((x - o) * 1e3, 1)),
                    (None, Some(_)) => format!("{name} new (none before)"),
                    (_, None) => format!("{name} not solved"),
                }
            })
            .collect();
        format!("References re-solved (the new F against the old E): {}.", words.join(", "))
    };
    out += " The audit history starts empty, so trust is rebuilt from new audits.";
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
            date: String::new(),
            review: [0; 3],
            not_migrated: Vec::new(),
            references: Vec::new(),
        }
    }

    /// The offer is spec §7.8's, verbatim: the species and the count from the predecessor, its folder, the estimate as
    /// "About 1 h 30 min", T_el 1 579 K, and the two buttons.
    #[test]
    fn migration_offer_text() {
        let m = offered();
        assert_eq!(
            offer_text(&m),
            "This collection was taught with the previous solver. Atomic Sandbox now fills electron levels with Fermi–Dirac smearing at 1 579 K, which fixes SCF stalls on loose clusters. Your collection \"H, N, O · 2 412 samples\" (models/collections/lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef) was taught with the old rule. Its samples need one re-solve with the new solver before the model uses them. Most will come back identical within solver noise. Open-shell and nearly degenerate ones change by up to a few mHa; those are checked against the rest before they rejoin the fit. About 1 h 30 min of GPU time. You can pause it and keep working. The original folder is kept unchanged, and older versions keep using it."
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
        assert_eq!(spaced(2412), "2 412");
        assert_eq!(spaced(412), "412");
        assert_eq!(spaced(1_234_567), "1 234 567");
        // Later folds it into one line, the count and the estimate kept
        assert_eq!(later_line(&m), "This collection was taught with the previous solver: 2 412 samples need one re-solve with the new solver (about 1 h 30 min of GPU time).");
    }

    /// The Activity line of spec §7.8, verbatim; no time left once there is none to say.
    #[test]
    fn migration_progress_line() {
        let m = MigrationStatus { done: 412, counts: [380, 22, 10, 0], eta_s: Some(52.0 * 60.0), state: MigrationPhase::Running, date: "2026-11-20".into(), ..offered() };
        assert_eq!(progress_line(&m), "Re-solving for the new solver: sample 412 of 2 412 · 380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated · ≈ 52 min left");
        assert_eq!(progress_line(&MigrationStatus { eta_s: None, ..m.clone() }), "Re-solving for the new solver: sample 412 of 2 412 · 380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated");
        let stopped = MigrationStatus { state: MigrationPhase::Paused, eta_s: Some(52.0 * 60.0), ..m };
        assert_eq!(stopped_line(&stopped), "Re-solving for the new solver stopped at sample 412 of 2 412 (380 identical · 22 changed only by −TS · 10 changed state · 0 not migrated). It goes on from there at the next start of the app, or now with Resume (about 52 min of GPU time left).");
    }

    /// The summary of spec §7.8: "identical within solver noise", "kept out", the old slug and the date; the review's
    /// waiting samples (stray atoms; Task 15) on their own clause; the not-migrated reasons in words; the references.
    #[test]
    fn migration_summary_text() {
        let m = MigrationStatus {
            done: 2412,
            counts: [2302, 64, 41, 5],
            eta_s: None,
            state: MigrationPhase::Reviewed,
            species: Vec::new(),
            date: "2026-11-20".into(),
            review: [35, 6, 0],
            not_migrated: vec![
                "the cold solve failed: SCF stalled: Δρ best 3.1e-3 not improved 2× in 50 iterations, still above the stall level 1.0e-3 (target 1.00e-6; stopped after 61 iterations)".into(),
                "box-affected: face density 3.10e-4 e/Bohr³ above the box rule's 1.00e-4 at h = 0.15".into(),
                "the cold solve failed: SCF stalled: Δρ best 2.0e-3 not improved 2× in 50 iterations, still above the stall level 1.0e-3 (target 1.00e-6; stopped after 55 iterations)".into(),
                "box-affected: face density 2.00e-4 e/Bohr³ above the box rule's 1.00e-4 at h = 0.15".into(),
                "box-affected: face density 1.20e-4 e/Bohr³ above the box rule's 1.00e-4 at h = 0.15".into(),
            ],
            references: vec![(1, 57, Some(-0.4791), Some(-0.4791)), (8, 57, Some(-15.8), Some(-15.8095))],
            ..offered()
        };
        let want = "Migrated from lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef on 2026-11-20:\n\
- 2 302 identical within solver noise (kept as they were);\n\
- 64 changed only by the electronic-entropy term −TS (re-labelled, kept in the fit);\n\
- 41 changed state (re-solved and reviewed: 35 rejoined the fit because the model without each predicts it within its bound; 6 are kept out, and the model falls back to DFT near them);\n\
- 5 not migrated (2 stalled, 3 now box-affected; they remain in the old collection).\n\
\n\
References re-solved (the new F against the old E): H +0.0 mHa, O −9.5 mHa. The audit history starts empty, so trust is rebuilt from new audits.";
        assert_eq!(summary_text(&m), want);
        let s = summary_text(&m);
        for words in ["identical within solver noise", "kept out", "lsda-h0.15-s0.15-tol1e-6-gpu-cores-0123456789abcdef"] {
            assert!(s.contains(words), "{words:?} in\n{s}");
        }
        assert!(!s.contains("discarded") && !s.contains("- 2 302 identical (") , "never 'discarded', never plain 'identical'");
        // the review's waiting samples (stray atoms, Task 15 review M1) and the ones it never judges get their clauses
        let waiting = MigrationStatus { state: MigrationPhase::Complete, review: [34, 5, 2], counts: [2302, 64, 41, 5], ..m.clone() };
        assert!(summary_text(&waiting).contains("- 41 changed state (re-solved and reviewed: 34 rejoined the fit because the model without each predicts it within its bound; 5 are kept out, and the model falls back to DFT near them; 2 waiting (stray atoms));"), "{}", summary_text(&waiting));
        let unjudged = MigrationStatus { review: [33, 5, 0], ..m.clone() };
        assert!(summary_text(&unjudged).contains("5 are kept out, and the model falls back to DFT near them; 3 not reviewed (they kept an earlier Forget or supersede, or wait for an isolated-atom reference));"), "{}", summary_text(&unjudged));
        // before the first review: what the review will do
        let before = MigrationStatus { state: MigrationPhase::Complete, review: [0; 3], ..m.clone() };
        assert!(summary_text(&before).contains("- 41 changed state (re-solved; the next refit reviews them: those the model without each predicts within its bound rejoin the fit, the others are kept out and the model falls back to DFT near them);"), "{}", summary_text(&before));
        // one of each: singular words; nothing not migrated; a reference that did not solve
        let one = MigrationStatus { counts: [1, 0, 1, 1], review: [0, 1, 0], not_migrated: vec!["solved at multiplicity 3, recorded 1".into()], references: vec![(8, 57, Some(-15.8), None)], ..m.clone() };
        let s = summary_text(&one);
        assert!(s.contains("- 1 changed state (re-solved and reviewed: 0 rejoined the fit because the model without each predicts it within its bound; 1 is kept out, and the model falls back to DFT near it);"), "{s}");
        assert!(s.contains("- 1 not migrated (1 solved at another multiplicity; it remains in the old collection)."), "{s}");
        assert!(s.contains("References re-solved (the new F against the old E): O not solved."), "{s}");
        let none = MigrationStatus { counts: [2, 0, 0, 0], review: [0; 3], not_migrated: Vec::new(), references: Vec::new(), ..m };
        let s = summary_text(&none);
        assert!(s.contains("- 0 not migrated.") && s.contains("References re-solved: none."), "{s}");
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
