//! `migration.txt`: the record of a collection migrated from a predecessor folder to a new solver
//! surface (the Fermi–Dirac switch; spec `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`
//! §7.3–7.6, plan Task 13). It lives in the **new** folder ([`crate::collection`]'s layout table);
//! the predecessor is only ever opened read-only (`Collection::open_read_only`). This module is the
//! file and its model; the pass that fills it is the provider's (`providers::sampling`, Task 14).
//! Atomic units throughout (Hartree, Bohr).
//!
//! # What it holds ([`MigrationLog`])
//!
//! - the source: its folder name (slug) and its `key.txt` text, the build that migrated and the
//!   date;
//! - the source `samples.bin`'s length and hash ([`source_fingerprint`]) when the pass started
//!   (F3): a resume whose source no longer matches (a purge, or an older build appending, since)
//!   starts again from sample 0 — the cursor names samples by index, and those indices moved;
//! - the cursor (the next source sample to re-solve) and the state (running, complete, reviewed);
//! - the counts per class — unchanged (U), entropy only (T), state changed (S) — and not migrated;
//! - every sample not migrated, by its source index, with its reason's code ([`NotMigrated`]) and the reason;
//! - the references re-solved: (Z, box n, the old energy, the new), either missing when there was
//!   none or it did not solve;
//! - class U's largest |ΔE| (Hartree) and |ΔF| (Hartree/Bohr), so the thresholds' margin is seen;
//! - the review log: per code-9 sample judged, its index in the **new** file, its leave-one-out
//!   ratio and whether it was released;
//! - the code-9 samples the review could not judge (stray atoms: no candidate fits them unmeasured),
//!   by their index in the new file — while any remain the state stays "complete";
//! - the **pending status** (Task 14 fix round 1, physics review I1): the index in the new file and
//!   the code of the sample being appended, written before its append whenever its code is not 0
//!   and cleared by the cursor write. A status record can only follow its sample record, so a crash
//!   between the two would leave the sample at code 0 (in the fit); the next attach applies the
//!   pending code first — only to the sample it names: its energy's bits and [`answer_print`] of its
//!   atoms, positions and forces must match (fix round 2), so a stale entry never lands on another;
//! - the superseded samples left without their superseder (fix round 1, M4): source indices of
//!   migrated code-2 samples whose likely superseder was not migrated.
//!
//! # The file
//!
//! UTF-8 text a person can read, one field per line, in this order:
//!
//! ```text
//! # Atomic Sandbox collection migration: …          (comment lines start with '#')
//! migration 1                                       (this file's format version)
//! source_slug <s>
//! source_key <k>                                    (then k lines, each two spaces and one line of key.txt)
//! build <s>
//! date <s>
//! source_len <u64>
//! source_hash <16 hex digits>
//! cursor <usize>
//! state running | complete | reviewed
//! counts unchanged <n> entropy_only <n> state_changed <n> not_migrated <n>
//! class_u_max <|ΔE|> <|ΔF|>
//! not_migrated <source index> <code> <reason>       (zero or more, in order; <code> a NotMigrated word)
//! reference <Z> <n> <old | -> <new | ->             (zero or more, in order)
//! review <new index> <ratio> released | kept        (zero or more, in order)
//! waiting <new index>                               (zero or more, in order: not judged, stray atoms)
//! pending <new index> <code> <energy bits> <print> (at most one; both 16 hex digits)
//! orphaned <source index>                           (zero or more, in order)
//! end
//! ```
//!
//! Numbers are written exactly: integers in decimal, f64 as Rust's shortest round-trip decimal
//! (bit for bit, −0.0 and non-finite values included). A string `<s>` is escaped onto its line
//! (`\\`, `\n`, `\r`), so a reason may hold any text. The reader is strict — every field once, in
//! any order, lists in file order, nothing unknown, the closing `end` present (a file cut short is
//! an error, never a silent default) — and refuses a newer `migration` version.
//!
//! **Format version 1 is not frozen yet**: no build with this file has shipped (the Fermi–Dirac switch is unreleased),
//! so its lines may still change at version 1 — as the `not_migrated` line did, gaining its code (plan Task 16 fix
//! round 1). Once a build ships, a change is a new `migration` version.
//!
//! # The summary
//!
//! [`summary_text`] is the migration's summary for people (spec §7.8): the Model tab shows it
//! (`sim::MigrationStatus::summary`), and the collection's `README.txt` carries it as its migration paragraph
//! (`Collection::write_readme`) — one text in both places.
//!
//! [`write_migration`] rewrites the file whole through `migration.txt.tmp`, synced and renamed into
//! place (on Windows with write-through), after every source sample's append (spec §7.3,
//! "Resume"): a crash leaves either the previous file or the new one, never a mix.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// The file's name in a collection's folder.
pub const MIGRATION_FILE: &str = "migration.txt";

/// The file's format version (the `migration` line; module doc).
pub const MIGRATION_FORMAT: u32 = 1;

/// Where a migration stands (spec §7.3, §7.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationState {
    /// The re-solve pass has samples left (resumed on the next attach or app start).
    Running,
    /// Every source sample was re-solved or listed as not migrated; code-9 reviews may be pending.
    Complete,
    /// Complete, and the code-9 samples were reviewed by a refit.
    Reviewed,
}

impl MigrationState {
    fn word(self) -> &'static str {
        match self {
            MigrationState::Running => "running",
            MigrationState::Complete => "complete",
            MigrationState::Reviewed => "reviewed",
        }
    }
}

/// The contents of `migration.txt` (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationLog {
    /// The predecessor folder's name under `models/collections`.
    pub source_slug: String,
    /// The predecessor's `key.txt`, as it is there.
    pub source_key: String,
    /// The build that migrated (its version).
    pub build: String,
    /// The day the migration started (YYYY-MM-DD).
    pub date: String,
    /// The source `samples.bin`'s length in bytes when the pass started (F3).
    pub source_len: u64,
    /// The source `samples.bin`'s hash when the pass started ([`source_fingerprint`]; F3).
    pub source_hash: u64,
    /// The next source sample index to re-solve.
    pub cursor: usize,
    pub state: MigrationState,
    /// Samples per class: unchanged (U), entropy only (T), state changed (S), not migrated.
    pub counts: [usize; 4],
    /// Each sample not migrated: its source index, its reason's code and the reason.
    pub not_migrated: Vec<(usize, NotMigrated, String)>,
    /// Each reference re-solved: (Z, grid points per side, the old energy, the new; Hartree).
    pub references: Vec<(u8, usize, Option<f64>, Option<f64>)>,
    /// Class U's largest |ΔE| (Hartree) and largest force-component |ΔF| (Hartree/Bohr).
    pub class_u_max: (f64, f64),
    /// The review log: (the sample's index in the new file, its leave-one-out ratio, released).
    pub review: Vec<(usize, f64, bool)>,
    /// The code-9 samples the last review could not judge (stray atoms), by their index in the new file.
    pub waiting: Vec<usize>,
    /// The status of the sample being appended (code ≠ 0), cleared by the cursor write.
    pub pending: Option<PendingStatus>,
    /// Source indices of migrated superseded (code 2) samples whose likely superseder was not migrated.
    pub orphaned: Vec<usize>,
}

/// Why a source sample was not migrated (spec §7.3 step 2): `migration.txt`'s `not_migrated` lines carry it as a word,
/// and the summary counts by it — never by matching the reason's prose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotMigrated {
    /// The cold solve's SCF stalled (the stall stop).
    Stalled,
    /// The cold solve hit its iteration cap.
    NotConverged,
    /// The cold solve failed otherwise.
    SolveFailed,
    /// The solve's face density is above the box rule.
    BoxAffected,
    /// The solve reported no face density.
    BoxUnknown,
    /// The solve carried a solver warning (core overlap, LSDA domain, Poisson cap).
    SolverWarning,
    /// The solve is not trained on (`EvalStats::untrainable`).
    Untrainable,
    /// The solve did not reproduce the recorded multiplicity.
    Multiplicity,
    /// Lone atoms: no neighbour within the model's cutoff.
    Lone,
    /// Its scene forced the multiplicity.
    ForcedMultiplicity,
    /// Its status code is not one a collection holds.
    StatusCode,
    /// Its scene could not be rebuilt (an isotope missing from the table).
    NoScene,
    /// Its scene is not on the collection's surface.
    OtherSurface,
}

impl NotMigrated {
    /// Every code, in the enum's order.
    pub const ALL: [NotMigrated; 13] = [
        NotMigrated::Stalled,
        NotMigrated::NotConverged,
        NotMigrated::SolveFailed,
        NotMigrated::BoxAffected,
        NotMigrated::BoxUnknown,
        NotMigrated::SolverWarning,
        NotMigrated::Untrainable,
        NotMigrated::Multiplicity,
        NotMigrated::Lone,
        NotMigrated::ForcedMultiplicity,
        NotMigrated::StatusCode,
        NotMigrated::NoScene,
        NotMigrated::OtherSurface,
    ];

    /// Its word in `migration.txt`.
    pub fn word(self) -> &'static str {
        match self {
            NotMigrated::Stalled => "stalled",
            NotMigrated::NotConverged => "not-converged",
            NotMigrated::SolveFailed => "solve-failed",
            NotMigrated::BoxAffected => "box-affected",
            NotMigrated::BoxUnknown => "box-unknown",
            NotMigrated::SolverWarning => "solver-warning",
            NotMigrated::Untrainable => "untrainable",
            NotMigrated::Multiplicity => "multiplicity",
            NotMigrated::Lone => "lone",
            NotMigrated::ForcedMultiplicity => "forced-multiplicity",
            NotMigrated::StatusCode => "status-code",
            NotMigrated::NoScene => "no-scene",
            NotMigrated::OtherSurface => "other-surface",
        }
    }

    /// The code of a `migration.txt` word.
    pub fn parse(w: &str) -> Option<NotMigrated> {
        NotMigrated::ALL.into_iter().find(|c| c.word() == w)
    }

    /// The summary's words for it ("2 stalled, 3 now box-affected").
    pub fn summary_words(self) -> &'static str {
        match self {
            NotMigrated::Stalled => "stalled",
            NotMigrated::NotConverged => "not converged",
            NotMigrated::SolveFailed => "failed to solve",
            NotMigrated::BoxAffected => "now box-affected",
            NotMigrated::BoxUnknown => "box status unknown",
            NotMigrated::SolverWarning => "with a solver warning",
            NotMigrated::Untrainable => "not trainable",
            NotMigrated::Multiplicity => "solved at another multiplicity",
            NotMigrated::Lone => "lone atoms",
            NotMigrated::ForcedMultiplicity => "with a forced multiplicity",
            NotMigrated::StatusCode => "with a status no collection holds",
            NotMigrated::NoScene => "whose scene could not be rebuilt",
            NotMigrated::OtherSurface => "on another surface",
        }
    }
}

/// "2 412" for 2412: thousands grouped as spec §7.8 prints them, with a narrow no-break space (U+202F), so a wrapping
/// label never breaks a number.
pub fn spaced(n: u64) -> String {
    let d = n.to_string();
    let mut out = String::with_capacity(d.len() + d.len());
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push('\u{202f}');
        }
        out.push(c);
    }
    out
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

/// "2 stalled, 3 now box-affected": the not-migrated samples by their code, in the order the codes first appear.
fn reasons(log: &MigrationLog) -> String {
    let mut tally: Vec<(NotMigrated, usize)> = Vec::new();
    for &(_, code, _) in &log.not_migrated {
        match tally.iter_mut().find(|(k, _)| *k == code) {
            Some(e) => e.1 += 1,
            None => tally.push((code, 1)),
        }
    }
    tally.iter().map(|(c, n)| format!("{} {}", spaced(*n as u64), c.summary_words())).collect::<Vec<_>>().join(", ")
}

/// The migration's summary (spec §7.8; module doc, "The summary"): the Model tab's and `README.txt`'s one text. `None`
/// while the pass has samples left (state running).
pub fn summary_text(log: &MigrationLog) -> Option<String> {
    if log.state == MigrationState::Running {
        return None;
    }
    let [u, t, s, n] = log.counts;
    let released = log.review.iter().filter(|e| e.2).count();
    let kept = log.review.len() - released;
    let waiting = log.waiting.len();
    let sp = |k: usize| spaced(k as u64);
    let date = if log.date.is_empty() { String::new() } else { format!(" on {}", log.date) };
    let mut out = format!("Migrated from {}{date}:\n", log.source_slug);
    out += &format!("- {} identical within solver noise (kept as they were);\n", sp(u));
    out += &format!("- {} changed only by the electronic-entropy term −TS (re-labelled, kept in the fit);\n", sp(t));
    if s == 0 {
        out += "- 0 changed state;\n";
    } else {
        let changed = if log.state == MigrationState::Complete && log.review.is_empty() && log.waiting.is_empty() {
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
    }
    out += &if n == 0 {
        "- 0 not migrated.\n".to_string()
    } else {
        let remain = if n == 1 { "it remains" } else { "they remain" };
        format!("- {} not migrated ({}; {remain} in the old collection).\n", sp(n), reasons(log))
    };
    out.push('\n');
    out += &if log.references.is_empty() {
        "References re-solved: none.".to_string()
    } else {
        let mut per_z: std::collections::BTreeMap<u8, usize> = Default::default();
        for r in &log.references {
            *per_z.entry(r.0).or_default() += 1;
        }
        let words: Vec<String> = log
            .references
            .iter()
            .map(|&(z, box_n, old, new)| {
                let sym = units::elements::symbol(z);
                let name = if per_z[&z] > 1 { format!("{sym} ({box_n}³)") } else { sym.to_string() };
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
    Some(out)
}

/// The status of a migrated sample being appended ([`MigrationLog::pending`]): its index in the new file, its code, and
/// its identity — the energy's bits and [`answer_print`] — so it is applied to that sample only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingStatus {
    pub index: usize,
    pub code: u8,
    pub energy_bits: u64,
    pub print: u64,
}

impl PendingStatus {
    /// The pending status of `s` at `index` with `code`.
    pub fn of(index: usize, code: u8, s: &crate::fit::Sample) -> PendingStatus {
        PendingStatus { index, code, energy_bits: s.energy.to_bits(), print: answer_print(s) }
    }

    /// Whether `s` is the sample this status names (its energy's bits and its [`answer_print`]).
    pub fn names(&self, s: &crate::fit::Sample) -> bool {
        s.energy.to_bits() == self.energy_bits && answer_print(s) == self.print
    }
}

/// FNV-1a 64 of a sample's atoms and the bits of its positions and forces: with the energy's bits, the identity of a
/// teacher answer (the collection's duplicate rule compares the same fields).
pub fn answer_print(s: &crate::fit::Sample) -> u64 {
    let mut b = Vec::with_capacity(s.z.len() * 49);
    b.extend_from_slice(&s.z);
    for v in s.pos.iter().chain(&s.forces) {
        for c in v {
            b.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    crate::model::fnv1a64(&b)
}

/// The length and the hash (64-bit FNV-1a of every byte) of a source `samples.bin` (F3: what
/// [`MigrationLog::source_len`] and [`MigrationLog::source_hash`] record, and what a resume compares).
pub fn source_fingerprint(samples: &Path) -> Result<(u64, u64), String> {
    let bytes = std::fs::read(samples).map_err(|e| format!("{}: {e}", samples.display()))?;
    Ok(fingerprint_bytes(&bytes))
}

/// [`source_fingerprint`] of bytes already read (`Collection::open_read_only` fingerprints the very
/// bytes it parsed).
pub fn fingerprint_bytes(bytes: &[u8]) -> (u64, u64) {
    (bytes.len() as u64, crate::model::fnv1a64(bytes))
}

/// `s` escaped onto one line (module doc).
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

fn unescape(s: &str) -> Result<String, String> {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            other => return Err(format!("'\\{}' is not an escape (\\\\, \\n, \\r)", other.map_or(String::new(), String::from))),
        }
    }
    Ok(out)
}

fn opt(x: Option<f64>) -> String {
    x.map_or("-".to_string(), |v| format!("{v:?}"))
}

/// The file's text (module doc).
fn encode(log: &MigrationLog) -> String {
    let mut t = String::new();
    t.push_str("# Atomic Sandbox collection migration: this collection's samples were re-solved from the folder below for the Fermi–Dirac solver.\n");
    t.push_str("# Classes: unchanged (kept as they were), entropy only (re-labelled by -TS), state changed (re-solved, code 9 until reviewed).\n");
    t.push_str(&format!("migration {MIGRATION_FORMAT}\n"));
    t.push_str(&format!("source_slug {}\n", escape(&log.source_slug)));
    let key: Vec<&str> = log.source_key.split('\n').collect();
    t.push_str(&format!("source_key {}\n", key.len()));
    for line in key {
        t.push_str(&format!("  {line}\n"));
    }
    t.push_str(&format!("build {}\n", escape(&log.build)));
    t.push_str(&format!("date {}\n", escape(&log.date)));
    t.push_str(&format!("source_len {}\n", log.source_len));
    t.push_str(&format!("source_hash {:016x}\n", log.source_hash));
    t.push_str(&format!("cursor {}\n", log.cursor));
    t.push_str(&format!("state {}\n", log.state.word()));
    let [u, tt, s, n] = log.counts;
    t.push_str(&format!("counts unchanged {u} entropy_only {tt} state_changed {s} not_migrated {n}\n"));
    t.push_str(&format!("class_u_max {:?} {:?}\n", log.class_u_max.0, log.class_u_max.1));
    for (i, code, why) in &log.not_migrated {
        t.push_str(&format!("not_migrated {i} {} {}\n", code.word(), escape(why)));
    }
    for &(z, n, old, new) in &log.references {
        t.push_str(&format!("reference {z} {n} {} {}\n", opt(old), opt(new)));
    }
    for &(i, ratio, released) in &log.review {
        t.push_str(&format!("review {i} {ratio:?} {}\n", if released { "released" } else { "kept" }));
    }
    for i in &log.waiting {
        t.push_str(&format!("waiting {i}\n"));
    }
    if let Some(p) = log.pending {
        t.push_str(&format!("pending {} {} {:016x} {:016x}\n", p.index, p.code, p.energy_bits, p.print));
    }
    for i in &log.orphaned {
        t.push_str(&format!("orphaned {i}\n"));
    }
    t.push_str("end\n");
    t
}

/// Reads [`encode`]'s text (module doc: strict).
fn decode(text: &str) -> Result<MigrationLog, String> {
    fn num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
        s.parse::<T>().map_err(|_| format!("{what} '{s}' is not a number of its kind"))
    }
    fn float(s: &str, what: &str) -> Result<f64, String> {
        num::<f64>(s, what)
    }
    fn opt_float(s: &str, what: &str) -> Result<Option<f64>, String> {
        if s == "-" {
            Ok(None)
        } else {
            float(s, what).map(Some)
        }
    }
    let mut lines = text.split('\n').enumerate();
    let (mut format, mut slug, mut key, mut build, mut date) = (None, None, None, None, None);
    let (mut len, mut hash, mut cursor, mut state, mut counts, mut u_max) = (None, None, None, None, None, None);
    let (mut not_migrated, mut references, mut review) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pending, mut orphaned, mut waiting) = (None, Vec::new(), Vec::new());
    let mut ended = false;
    let once = |slot: bool, name: &str| if slot { Err(format!("'{name}' appears twice")) } else { Ok(()) };
    while let Some((k, raw)) = lines.next() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if ended {
            if line.trim().is_empty() {
                continue;
            }
            return Err(format!("line {}: text after 'end'", k + 1));
        }
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |e: String| format!("line {}: {e}", k + 1);
        let (name, value) = line.split_once(' ').unwrap_or((line, ""));
        let fields: Vec<&str> = value.split(' ').collect();
        match name {
            "migration" => {
                once(format.is_some(), name).map_err(at)?;
                let v: u32 = num(value, "the format version").map_err(at)?;
                if v > MIGRATION_FORMAT {
                    return Err(at(format!("migration format {v} was written by a newer build (this build reads 1 to {MIGRATION_FORMAT})")));
                }
                format = Some(v);
            }
            "source_slug" => {
                once(slug.is_some(), name).map_err(at)?;
                slug = Some(unescape(value).map_err(at)?);
            }
            "source_key" => {
                once(key.is_some(), name).map_err(at)?;
                let n: usize = num(value, "the key's line count").map_err(at)?;
                let mut body = Vec::with_capacity(n);
                for _ in 0..n {
                    let (j, l) = lines.next().ok_or_else(|| at("the key ends early".into()))?;
                    body.push(l.strip_prefix("  ").ok_or_else(|| format!("line {}: a key line starts with two spaces", j + 1))?);
                }
                key = Some(body.join("\n"));
            }
            "build" => {
                once(build.is_some(), name).map_err(at)?;
                build = Some(unescape(value).map_err(at)?);
            }
            "date" => {
                once(date.is_some(), name).map_err(at)?;
                date = Some(unescape(value).map_err(at)?);
            }
            "source_len" => {
                once(len.is_some(), name).map_err(at)?;
                len = Some(num::<u64>(value, "source_len").map_err(at)?);
            }
            "source_hash" => {
                once(hash.is_some(), name).map_err(at)?;
                if value.len() != 16 {
                    return Err(at(format!("source_hash '{value}' is not 16 hex digits")));
                }
                hash = Some(u64::from_str_radix(value, 16).map_err(|_| at(format!("source_hash '{value}' is not 16 hex digits")))?);
            }
            "cursor" => {
                once(cursor.is_some(), name).map_err(at)?;
                cursor = Some(num::<usize>(value, "cursor").map_err(at)?);
            }
            "state" => {
                once(state.is_some(), name).map_err(at)?;
                state = Some(match value {
                    "running" => MigrationState::Running,
                    "complete" => MigrationState::Complete,
                    "reviewed" => MigrationState::Reviewed,
                    other => return Err(at(format!("state '{other}' is not running, complete or reviewed"))),
                });
            }
            "counts" => {
                once(counts.is_some(), name).map_err(at)?;
                let [l0, u, l1, t, l2, s, l3, n] = fields[..] else { return Err(at("counts are 'unchanged U entropy_only T state_changed S not_migrated N'".into())) };
                if [l0, l1, l2, l3] != ["unchanged", "entropy_only", "state_changed", "not_migrated"] {
                    return Err(at("counts are 'unchanged U entropy_only T state_changed S not_migrated N'".into()));
                }
                counts = Some([num(u, "a count").map_err(at)?, num(t, "a count").map_err(at)?, num(s, "a count").map_err(at)?, num(n, "a count").map_err(at)?]);
            }
            "class_u_max" => {
                once(u_max.is_some(), name).map_err(at)?;
                let [e, f] = fields[..] else { return Err(at("class_u_max is '|dE| |dF|'".into())) };
                u_max = Some((float(e, "|dE|").map_err(at)?, float(f, "|dF|").map_err(at)?));
            }
            "not_migrated" => {
                let (i, rest) = value.split_once(' ').unwrap_or((value, ""));
                let (code, why) = rest.split_once(' ').unwrap_or((rest, ""));
                let code = NotMigrated::parse(code).ok_or_else(|| at(format!("'{code}' is not a not-migrated code")))?;
                not_migrated.push((num::<usize>(i, "a sample index").map_err(at)?, code, unescape(why).map_err(at)?));
            }
            "reference" => {
                let [z, n, old, new] = fields[..] else { return Err(at("a reference is 'Z n old new'".into())) };
                let z: u8 = num(z, "Z").map_err(at)?;
                if z == 0 {
                    return Err(at("Z 0".into()));
                }
                references.push((z, num::<usize>(n, "n").map_err(at)?, opt_float(old, "the old energy").map_err(at)?, opt_float(new, "the new energy").map_err(at)?));
            }
            "review" => {
                let [i, ratio, verdict] = fields[..] else { return Err(at("a review is 'index ratio released|kept'".into())) };
                let released = match verdict {
                    "released" => true,
                    "kept" => false,
                    other => return Err(at(format!("'{other}' is not released or kept"))),
                };
                review.push((num::<usize>(i, "a sample index").map_err(at)?, float(ratio, "the ratio").map_err(at)?, released));
            }
            "pending" => {
                once(pending.is_some(), name).map_err(at)?;
                let [k, code, e, print] = fields[..] else { return Err(at("a pending status is 'index code energy-bits print'".into())) };
                let hex = |w: &str| if w.len() == 16 { u64::from_str_radix(w, 16).ok() } else { None };
                let (Some(energy_bits), Some(print)) = (hex(e), hex(print)) else { return Err(at("a pending status's energy bits and print are 16 hex digits".into())) };
                pending = Some(PendingStatus { index: num::<usize>(k, "a sample index").map_err(at)?, code: num::<u8>(code, "a status code").map_err(at)?, energy_bits, print });
            }
            "orphaned" => orphaned.push(num::<usize>(value, "a sample index").map_err(at)?),
            "waiting" => waiting.push(num::<usize>(value, "a sample index").map_err(at)?),
            "end" if value.is_empty() => ended = true,
            other => return Err(at(format!("unknown line '{other}'"))),
        }
    }
    if !ended {
        return Err("the file ends without its closing 'end' (cut short?)".into());
    }
    let need = |what: &str| format!("'{what}' is missing");
    format.ok_or_else(|| need("migration"))?;
    Ok(MigrationLog {
        source_slug: slug.ok_or_else(|| need("source_slug"))?,
        source_key: key.ok_or_else(|| need("source_key"))?,
        build: build.ok_or_else(|| need("build"))?,
        date: date.ok_or_else(|| need("date"))?,
        source_len: len.ok_or_else(|| need("source_len"))?,
        source_hash: hash.ok_or_else(|| need("source_hash"))?,
        cursor: cursor.ok_or_else(|| need("cursor"))?,
        state: state.ok_or_else(|| need("state"))?,
        counts: counts.ok_or_else(|| need("counts"))?,
        not_migrated,
        references,
        class_u_max: u_max.ok_or_else(|| need("class_u_max"))?,
        review,
        waiting,
        pending,
        orphaned,
    })
}

/// Reads `dir/migration.txt` (module doc): `Ok(None)` when there is none (the folder was not made
/// by a migration), `Err` for a file that cannot be read or does not parse.
pub fn read_migration(dir: &Path) -> Result<Option<MigrationLog>, String> {
    let path = dir.join(MIGRATION_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => decode(&text).map(Some).map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

#[cfg(test)]
thread_local! {
    /// Test hook: the next [`write_migration`] on this thread stops after the temporary file is
    /// written and synced, before the rename (a crash there).
    static FAIL_BEFORE_RENAME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Writes `log` to `dir/migration.txt` through `migration.txt.tmp`, synced and renamed into place
/// (module doc): a crash leaves the previous file whole.
pub fn write_migration(dir: &Path, log: &MigrationLog) -> Result<(), String> {
    let path = dir.join(MIGRATION_FILE);
    let tmp = dir.join(format!("{MIGRATION_FILE}.tmp"));
    let written = (|| -> io::Result<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(encode(log).as_bytes())?;
        f.sync_all()
    })();
    written.map_err(|e| format!("{}: {e}", tmp.display()))?;
    #[cfg(test)]
    if FAIL_BEFORE_RENAME.with(|f| f.replace(false)) {
        return Err(format!("{}: injected crash before the rename", path.display()));
    }
    crate::collection::replace_file(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A fresh directory of a test's own, removed when dropped.
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn temp(name: &str) -> TempDir {
        let d = std::env::temp_dir().join(format!("asmigr-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        TempDir(d)
    }

    /// A log with awkward values: a multi-line key, a reason with a newline, a backslash and a
    /// trailing space, 17-digit floats, −0.0, a missing old and a missing new reference, the
    /// largest u64 hash.
    fn log() -> MigrationLog {
        let key = crate::model::SurfaceKey { functional: "lsda".into(), h: 0.15, sigma: 0.15, tol: 1e-6, teacher: "Kohn-Sham (GPU)".into(), ionic: "accepted/0123456789abcdef".into() };
        MigrationLog {
            source_slug: key.slug(),
            source_key: key.to_text(),
            build: "0.9.3".into(),
            date: "2026-11-20".into(),
            source_len: 1_234_567_890_123,
            source_hash: u64::MAX,
            cursor: 412,
            state: MigrationState::Running,
            counts: [380, 22, 10, 2],
            not_migrated: vec![(7, NotMigrated::Stalled, "stalled".into()), (19, NotMigrated::BoxAffected, "box-affected: face density 3.1e-4 e/Bohr³ above 1.0e-4\nsecond line \\ tail ".into())],
            references: vec![(1, 57, Some(-0.478_123_456_789_012_3), Some(-0.479)), (8, 57, None, Some(-15.9)), (8, 81, Some(-15.8), None)],
            class_u_max: (9.876_543_210_987_654e-6, -0.0),
            review: vec![(1700, 0.25, true), (1701, 3.5, false), (2400, 1.0, true)],
            waiting: vec![1703, 2401],
            pending: Some(PendingStatus { index: 1702, code: 9, energy_bits: (-2.25f64).to_bits(), print: 0x0123_4567_89ab_cdef }),
            orphaned: vec![3, 17],
        }
    }

    /// Spec §7.5: every field round-trips exactly (floats bit for bit, strings byte for byte, the
    /// source fingerprint F3 included), in each state; an absent file reads as `None`; the file is
    /// text a person can read.
    #[test]
    fn migration_log_round_trips() {
        let d = temp("round-trip");
        assert_eq!(read_migration(&d.0).unwrap(), None, "no migration.txt: no migration");
        for state in [MigrationState::Running, MigrationState::Complete, MigrationState::Reviewed] {
            let want = MigrationLog { state, ..log() };
            write_migration(&d.0, &want).unwrap();
            let back = read_migration(&d.0).unwrap().expect("written");
            assert_eq!(back, want);
            assert_eq!((back.source_len, back.source_hash), (want.source_len, want.source_hash), "F3: the source fingerprint");
            assert_eq!(back.class_u_max.0.to_bits(), want.class_u_max.0.to_bits());
            assert_eq!(back.class_u_max.1.to_bits(), (-0.0f64).to_bits(), "−0.0 kept");
            assert_eq!(back.references[0].2.unwrap().to_bits(), want.references[0].2.unwrap().to_bits());
        }
        // empty lists and an empty key round-trip too
        let bare = MigrationLog { not_migrated: Vec::new(), references: Vec::new(), review: Vec::new(), waiting: Vec::new(), source_key: String::new(), pending: None, orphaned: Vec::new(), ..log() };
        write_migration(&d.0, &bare).unwrap();
        assert_eq!(read_migration(&d.0).unwrap(), Some(bare));
        // readable: one line per field, the key's lines as they are in key.txt
        write_migration(&d.0, &log()).unwrap();
        let text = std::fs::read_to_string(d.0.join(MIGRATION_FILE)).unwrap();
        for line in ["state running", "cursor 412", "functional lsda", "not_migrated 7 stalled stalled", "waiting 2401", "pending 1702 9 c002000000000000 0123456789abcdef", "orphaned 17"] {
            assert!(text.lines().any(|l| l.trim_start() == line), "{line:?} in\n{text}");
        }
        // a damaged file is an error, never a silent default
        for (what, from, to) in [("a bad state", "state running", "state paused"), ("a bad count", "cursor 412", "cursor -1"), ("an unknown line", "cursor 412", "cursor 412\nspeed 3"), ("an unknown not-migrated code", "not_migrated 7 stalled stalled", "not_migrated 7 bogus stalled")] {
            std::fs::write(d.0.join(MIGRATION_FILE), text.replacen(from, to, 1)).unwrap();
            assert!(read_migration(&d.0).is_err(), "{what}");
        }
        let cut = &text[..text.len() / 2];
        std::fs::write(d.0.join(MIGRATION_FILE), cut).unwrap();
        assert!(read_migration(&d.0).is_err(), "a file cut short");
    }

    /// F3: the source fingerprint is the file's length and its FNV-1a hash: the same bytes give the
    /// same pair, one appended or changed byte another; a missing file is an error.
    #[test]
    fn source_fingerprint_follows_the_file() {
        let d = temp("fingerprint");
        let p = d.0.join("samples.bin");
        std::fs::write(&p, b"ASCOLL1\0abc").unwrap();
        let a = source_fingerprint(&p).unwrap();
        assert_eq!(a, (11, crate::model::fnv1a64(b"ASCOLL1\0abc")));
        assert_eq!(source_fingerprint(&p).unwrap(), a);
        std::fs::write(&p, b"ASCOLL1\0abd").unwrap();
        let b = source_fingerprint(&p).unwrap();
        assert_eq!(b.0, a.0);
        assert_ne!(b.1, a.1, "a changed byte");
        std::fs::write(&p, b"ASCOLL1\0abcd").unwrap();
        assert_ne!(source_fingerprint(&p).unwrap(), a, "an appended byte");
        assert!(source_fingerprint(&d.0.join("absent")).is_err());
    }

    /// Spec §7.3 "Resume": `migration.txt` is rewritten through a temporary file renamed into
    /// place, so a crash mid-write (injected here: the temporary is written, the rename never
    /// happens) leaves the previous file whole, and the next write succeeds.
    #[test]
    fn migration_log_write_is_atomic() {
        let d = temp("atomic");
        let first = log();
        write_migration(&d.0, &first).unwrap();
        let before = std::fs::read(d.0.join(MIGRATION_FILE)).unwrap();
        let next = MigrationLog { cursor: 413, counts: [381, 22, 10, 2], ..log() };
        FAIL_BEFORE_RENAME.with(|f| f.set(true));
        assert!(write_migration(&d.0, &next).is_err(), "the injected crash");
        assert_eq!(std::fs::read(d.0.join(MIGRATION_FILE)).unwrap(), before, "the previous file whole");
        assert_eq!(read_migration(&d.0).unwrap(), Some(first));
        write_migration(&d.0, &next).unwrap();
        assert_eq!(read_migration(&d.0).unwrap(), Some(next));
    }
}
