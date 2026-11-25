//! The builder's search box (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md`
//! §4 Goal 2, §7, §8): a name, a formula or a PubChem CID typed at the top of the palette column
//! lists the matching molecules; picking one stamps it onto the canvas as a tidy 2D drawing (the
//! record's own layout, every hydrogen explicit) beside what is drawn, as one undo step.
//!
//! **The query** ([`parse_query`]) is compared after [`fold`]: Unicode subscripts read as digits
//! (`H₂O` is `H2O`), superscript charges as `^` and the charge (`OH⁻` is `OH^-`), whitespace runs
//! as one space. All digits is a CID (`000702` is 702); a formula written as chemists write it
//! (`C2H5OH`, `NH4+`) is a formula, and so is an ASCII one in the wrong case whose letters split
//! into the builder's element symbols (Z ≤ 18), a two-letter symbol tried before a one-letter one
//! (`h2o` → `H2O`, `cl2` → `Cl2`, `nacl` → `NaCl`, `nh4+` → `NH4+`, `co` → `CO`); anything else is
//! a name.
//!
//! **Local results** ([`search_local`]): the shipped list, then the molecule cache (a cached
//! record the shipped list holds by CID is left out), ranked: the exact name (or the CID), a name
//! starting with the query, an exact synonym, a synonym starting with it, the formula (the same
//! element counts, and the same net charge when the query gives one), then — for 3 characters or
//! more — a name or synonym containing it. Case-insensitive; at most [`MAX_RESULTS`]; within a rank
//! the shorter name first.
//!
//! **Online** (spec §7): "Search PubChem for '…'" is offered only with the builder's online setting
//! on and PubChem reachable ([`super::online::Session::can_search`]: [`super::online::network_allowed`] for the
//! app's transport, so never in a test build, a replay or a screenshot), and only the user's
//! click (or Enter with no local result) asks it, and only while no local result is an exact match
//! ([`is_exact`]). A name goes to PubChem's name endpoint as folded, a formula in the user's case
//! but plain ASCII (not the `^` notation, [`online_query`]), a CID to its compound; the answer is
//! listed below the local results (duplicates by CID left out, in place of the lowest-ranked local
//! rows: the whole list at most [`MAX_RESULTS`], [`combined_rows`]), and a failure is one short
//! line, never a toast. The setting
//! going off stops a search in flight before its next request ([`Builder::poll_search`]). A name holding `/` is sent
//! percent-encoded (`%2F`) in the path segment, which PubChem often refuses: that is a 4xx, read as
//! "no results on PubChem" (PUG REST's POST form would carry it, but the transport is GET-only).
//!
//! **Rule 1 (spec §3).** A record only places atoms on the canvas: it never leaves the builder.
use super::online::{Cache, NetError};
use super::records::{hill, Library, Record, Source};
use super::{Atom, Bond, Builder};
use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};
use crate::ui::ptable::Z_ENABLED;
use units::elements::{symbol, z_of_symbol};

/// The most local results the box lists (spec §8).
pub const MAX_RESULTS: usize = 8;

/// The longest name or formula (bytes, as folded) the online search sends.
pub const MAX_NAME: usize = 200;

/// The gap (canvas units) between the drawing's bounding box and a stamped molecule's.
pub const STAMP_GAP: f32 = 1.5;

/// How long the box shows "searching PubChem…" before it reads as not answered (the worker's
/// search has [`super::online::TIMEOUT`], and may queue behind a drawn fragment's lookup); an answer
/// arriving later still replaces it.
pub const ONLINE_WAIT: Duration = Duration::from_secs(12);

/// What a search box query asks for ([`parse_query`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Query {
    Empty,
    Cid(u64),
    /// The formula as folded, its letters' case corrected (`h2o` → `H2O`).
    Formula(String),
    /// The name as folded.
    Name(String),
}

/// One search result.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub name: String,
    /// The Hill formula ([`hill`]).
    pub formula: String,
    /// "shipped", "cached" or "PubChem".
    pub source: &'static str,
    pub record: Option<Record>,
}

impl Hit {
    fn of(r: &Record, source: &'static str) -> Hit {
        Hit { name: r.name.clone(), formula: hill(&r.z), source, record: Some(r.clone()) }
    }

    /// The PubChem compound the hit is, if it is one.
    pub fn cid(&self) -> Option<u64> {
        match self.record.as_ref()?.source {
            Source::PubChem { cid } => Some(cid),
            Source::Lira { .. } => None,
        }
    }
}

/// The query as compared (module doc): subscript digits as digits, a run of superscript digits
/// and signs as `^` and their ASCII (`SO₄²⁻` → `SO4^2-`), U+2212 as `-`, control characters
/// dropped, whitespace runs as one space, trimmed.
pub fn fold(q: &str) -> String {
    fold_marked(q, true)
}

/// [`fold`] with the superscripts as plain ASCII and no `^`, typed or not (`SO₄²⁻`, `SO4^2-` → `SO42-`): what a formula query
/// sends to PubChem.
pub fn plain(q: &str) -> String {
    fold_marked(q, false)
}

fn fold_marked(q: &str, caret: bool) -> String {
    let mut out = String::with_capacity(q.len());
    let (mut sup, mut space) = (false, false);
    for c in q.chars() {
        let (ch, is_sup) = match c {
            '\u{2080}'..='\u{2089}' => (char::from(b'0' + (c as u32 - 0x2080) as u8), false),
            '\u{2070}' => ('0', true),
            '\u{00B9}' => ('1', true),
            '\u{00B2}' => ('2', true),
            '\u{00B3}' => ('3', true),
            '\u{2074}'..='\u{2079}' => (char::from(b'4' + (c as u32 - 0x2074) as u8), true),
            '\u{207A}' => ('+', true),
            '\u{207B}' => ('-', true),
            '\u{2212}' => ('-', false),
            c if c.is_whitespace() => {
                space = true;
                continue;
            }
            c if c.is_control() => continue,
            // a typed caret is our notation, never sent (plain ASCII has none)
            '^' if !caret => continue,
            c => (c, false),
        };
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        if caret && is_sup && !sup {
            out.push('^');
        }
        sup = is_sup;
        out.push(ch);
    }
    out
}

/// A formula as written: its element counts, its charge when given, how many element tokens it
/// has and whether any carries a count.
#[derive(Debug, PartialEq)]
struct Formula {
    counts: BTreeMap<u8, u32>,
    charge: Option<i32>,
    tokens: usize,
    digits: bool,
}

/// Reads a folded formula, case-sensitively: element symbols (`C`, `Cl`) each with an optional
/// count 1–999, then optionally the charge (`+`, `--`, `2-`, `+2`, after an optional `^`).
fn formula(s: &str) -> Option<Formula> {
    let b = s.as_bytes();
    let mut f = Formula { counts: BTreeMap::new(), charge: None, tokens: 0, digits: false };
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_uppercase() {
        let start = i;
        i += 1;
        if i < b.len() && b[i].is_ascii_lowercase() {
            i += 1;
        }
        let sym = &s[start..i];
        let z = z_of_symbol(sym).filter(|&z| symbol(z) == sym)?;
        let d = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let n = if i > d {
            f.digits = true;
            s[d..i].parse::<u32>().ok().filter(|n| (1..=999).contains(n))?
        } else {
            1
        };
        *f.counts.entry(z).or_default() += n;
        f.tokens += 1;
    }
    if f.tokens == 0 {
        return None;
    }
    let rest = &s[i..];
    let (caret, rest) = rest.strip_prefix('^').map_or((false, rest), |r| (true, r));
    f.charge = match charge(rest) {
        Some(q) => Some(q),
        None if rest.is_empty() && !caret => None,
        None => return None,
    };
    Some(f)
}

/// A charge suffix: `+`, `+++`, `-`, `2+`, `+2` (1–9); None for anything else.
fn charge(s: &str) -> Option<i32> {
    let sign = |c: char| match c {
        '+' => Some(1),
        '-' => Some(-1),
        _ => None,
    };
    let mut cs = s.chars();
    let first = cs.next()?;
    if let Some(sg) = sign(first) {
        let tail = cs.as_str();
        if tail.chars().all(|c| c == first) {
            return Some(sg * (1 + tail.len() as i32)).filter(|q| q.abs() <= 9);
        }
        return tail.parse::<i32>().ok().filter(|n| (1..=9).contains(n)).map(|n| sg * n);
    }
    let last = s.chars().last()?;
    let digits = &s[..s.len() - last.len_utf8()];
    let n = digits.parse::<i32>().ok().filter(|n| (1..=9).contains(n))?;
    sign(last).map(|sg| sg * n)
}

/// [`formula`], else (an ASCII query) the same with each run of letters split into element symbols
/// whatever their case ([`symbols`]: `nacl` → `NaCl`, `h2o` → `H2O`): the formula's text (case
/// corrected) and its reading.
fn formula_any_case(s: &str) -> Option<(String, Formula)> {
    if let Some(f) = formula(s) {
        return Some((s.to_string(), f));
    }
    if !s.is_ascii() {
        return None;
    }
    let b = s.as_bytes();
    let mut fixed = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        let start = i;
        if b[i].is_ascii_alphabetic() {
            while i < b.len() && b[i].is_ascii_alphabetic() {
                i += 1;
            }
            fixed += &symbols(&s[start..i])?;
        } else {
            fixed.push(b[i] as char);
            i += 1;
        }
    }
    formula(&fixed).map(|f| (fixed, f))
}

/// A run of ASCII letters split into element symbols of the builder's range (Z 1–[`Z_ENABLED`]: no
/// record holds others), case-insensitively, a two-letter symbol tried before a one-letter one and
/// the other tried when the rest does not split (`nacl` → `NaCl`, `cl` → `Cl`, `nh` → `NH`, not
/// nihonium; `co` → `CO`, not cobalt); None when no split exists. Linear: a position that failed
/// once is not tried again.
fn symbols(run: &str) -> Option<String> {
    fn go(run: &str, i: usize, failed: &mut [bool], out: &mut Vec<&'static str>) -> bool {
        if i == run.len() {
            return true;
        }
        if failed[i] {
            return false;
        }
        for len in [2, 1] {
            if let Some(z) = run.get(i..i + len).and_then(z_of_symbol).filter(|&z| z <= Z_ENABLED) {
                out.push(symbol(z));
                if go(run, i + len, failed, out) {
                    return true;
                }
                out.pop();
            }
        }
        failed[i] = true;
        false
    }
    let mut out = Vec::new();
    go(run, 0, &mut vec![false; run.len()], &mut out).then(|| out.concat())
}

/// What the box's text asks for (module doc). Never panics, whatever the text.
pub fn parse_query(q: &str) -> Query {
    let s = fold(q);
    if s.is_empty() {
        return Query::Empty;
    }
    if s.bytes().all(|b| b.is_ascii_digit()) {
        return match s.parse::<u64>() {
            Ok(c) if c > 0 => Query::Cid(c),
            _ => Query::Name(s),
        };
    }
    match formula_any_case(&s) {
        // one element alone ("He", "Co") reads as a name; a count, two elements or a charge make a formula
        Some((text, f)) if f.digits || f.tokens >= 2 || f.charge.is_some() => Query::Formula(text),
        _ => Query::Name(s),
    }
}

/// The shipped list's and the cache's matches for `q` (module doc's ranking), at most
/// [`MAX_RESULTS`].
pub fn search_local(q: &str, lib: &Library, cache: &Cache) -> Vec<Hit> {
    let s = fold(q);
    if s.is_empty() {
        return Vec::new();
    }
    let key = s.to_lowercase();
    let cid = match parse_query(q) {
        Query::Cid(c) => Some(c),
        _ => None,
    };
    let f = formula_any_case(&s).map(|(_, f)| f);
    let mut shipped_cids: HashSet<u64> = HashSet::new();
    for r in lib.records() {
        if let Source::PubChem { cid } = r.source {
            shipped_cids.insert(cid);
        }
    }
    let cached = cache.all();
    let candidates = lib.records().iter().map(|r| (r, 0u8, "shipped")).chain(
        cached
            .iter()
            .filter(|r| match r.source {
                Source::PubChem { cid } => !shipped_cids.contains(&cid),
                Source::Lira { .. } => true,
            })
            .map(|r| (r, 1u8, "cached")),
    );
    let mut ranked: Vec<(u8, usize, u8, &str, &Record, &'static str)> = candidates
        .filter_map(|(r, from, source)| rank(r, &key, cid, f.as_ref()).map(|t| (t, r.name.chars().count(), from, r.name.as_str(), r, source)))
        .collect();
    ranked.sort_by(|a, b| (a.0, a.1, a.2, a.3).cmp(&(b.0, b.1, b.2, b.3)));
    ranked.into_iter().take(MAX_RESULTS).map(|(.., r, source)| Hit::of(r, source)).collect()
}

/// A record's rank for the query (lower first), None when it does not match.
fn rank(r: &Record, key: &str, cid: Option<u64>, f: Option<&Formula>) -> Option<u8> {
    let name = r.name.to_lowercase();
    let syns: Vec<String> = r.synonyms.iter().map(|s| s.to_lowercase()).collect();
    let is_cid = matches!((cid, &r.source), (Some(c), Source::PubChem { cid }) if c == *cid);
    if is_cid || name == key {
        return Some(0);
    }
    if name.starts_with(key) {
        return Some(1);
    }
    if syns.iter().any(|s| s == key) {
        return Some(2);
    }
    if syns.iter().any(|s| s.starts_with(key)) {
        return Some(3);
    }
    if let Some(f) = f {
        let mut counts: BTreeMap<u8, u32> = BTreeMap::new();
        for &z in &r.z {
            *counts.entry(z).or_default() += 1;
        }
        let q: i32 = r.charge.iter().map(|&q| q as i32).sum();
        if counts == f.counts && f.charge.is_none_or(|c| c == q) {
            return Some(4);
        }
    }
    if key.chars().count() >= 3 && (name.contains(key) || syns.iter().any(|s| s.contains(key))) {
        return Some(5);
    }
    None
}

/// The bounding box of some canvas points (None for none).
fn bounds(p: impl Iterator<Item = [f32; 2]>) -> Option<([f32; 2], [f32; 2])> {
    p.fold(None, |acc, q| {
        let (lo, hi) = acc.unwrap_or((q, q));
        Some(([lo[0].min(q[0]), lo[1].min(q[1])], [hi[0].max(q[0]), hi[1].max(q[1])]))
    })
}

/// What the online search sends for the box's text: as [`parse_query`] reads it, except that a
/// formula goes as the user wrote it — its case kept, but in plain ASCII ([`plain`]: `H₂O` →
/// `H2O`, `SO₄²⁻` → `SO42-`, never our `^` notation; PubChem's synonyms are written so).
pub fn online_query(q: &str) -> Query {
    match parse_query(q) {
        Query::Formula(_) => Query::Formula(plain(q)),
        other => other,
    }
}

/// A local hit answers the query exactly (rank 0: its name, or its CID): then there is nothing to
/// ask PubChem for.
pub fn is_exact(q: &str, h: &Hit) -> bool {
    let key = fold(q).to_lowercase();
    !key.is_empty() && (h.name.to_lowercase() == key || matches!((parse_query(q), h.cid()), (Query::Cid(c), Some(d)) if c == d))
}

/// The list the box shows once PubChem answered (spec §8: at most [`MAX_RESULTS`] rows): PubChem's
/// hits whose CID the local list does not hold, in place of the lowest-ranked local rows; the local
/// rows kept, then PubChem's.
pub fn combined_rows<'a>(local: &'a [Hit], online: &'a [Hit]) -> (Vec<&'a Hit>, Vec<&'a Hit>) {
    let seen: Vec<u64> = local.iter().filter_map(Hit::cid).collect();
    let fresh: Vec<&Hit> = online.iter().filter(|h| h.cid().is_none_or(|c| !seen.contains(&c))).take(MAX_RESULTS).collect();
    let keep = MAX_RESULTS - fresh.len();
    (local.iter().take(keep).collect(), fresh)
}

/// The PubChem search asked for one query: its id, when, and how it went.
pub(crate) struct Online {
    /// The query's [`fold`], lowercased: the box's text it belongs to.
    query: String,
    id: u64,
    asked: Instant,
    state: OnlineState,
    /// No answer yet. After [`ONLINE_WAIT`] the box reads it as not answered, but the answer may
    /// still come: the window repaints slowly for it until [`super::online::LATE_WAIT`].
    unanswered: bool,
}

enum OnlineState {
    Waiting,
    Done(Vec<Hit>),
    Failed(NetError),
}

/// The search box's results: the local ones for the text they were made for, and the online search.
#[derive(Default)]
pub(crate) struct SearchBox {
    local: Option<(String, Vec<Hit>)>,
    online: Option<Online>,
    /// The list searched in place of the shipped one (tests, so that they keep to the records they
    /// name whatever the shipped list holds).
    #[cfg(test)]
    pub(crate) library: Option<&'static Library>,
}

fn online_key(q: &str) -> String {
    fold(q).to_lowercase()
}

impl Builder {
    /// Stamps a record onto the canvas as one undo step (spec §4 Goal 2): its atoms at its 2D
    /// layout (canvas units, mean bond 1), moved right of the drawing's bounding box by
    /// [`STAMP_GAP`] and level with its middle — centred on the canvas origin when nothing is
    /// drawn — with its bonds (Kekulé orders) and formal charges. The stamped atoms become the
    /// selection (to move them as one), and the view centres on the whole drawing. A record whose
    /// lists disagree is ignored.
    pub fn stamp_record(&mut self, r: &Record) {
        let n = r.z.len();
        if n == 0 || r.xy.len() != n || r.charge.len() != n || r.bonds.iter().any(|&(a, b, _)| a >= n || b >= n) {
            return;
        }
        let Some((lo, hi)) = bounds(r.xy.iter().copied()) else { return };
        let shift = match bounds(self.drawing.atoms.iter().map(|a| a.pos)) {
            None => [-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0],
            Some((dlo, dhi)) => [dhi[0] + STAMP_GAP - lo[0], (dlo[1] + dhi[1]) / 2.0 - (lo[1] + hi[1]) / 2.0],
        };
        self.checkpoint();
        let base = self.drawing.atoms.len();
        for i in 0..n {
            self.drawing.atoms.push(Atom { z: r.z[i], pos: [r.xy[i][0] + shift[0], r.xy[i][1] + shift[1]], charge: r.charge[i] });
        }
        for &(a, b, order) in &r.bonds {
            self.drawing.bonds.push(Bond { a: base + a, b: base + b, order });
        }
        self.forget_indices();
        self.selection = (base..base + n).collect();
        if let Some((lo, hi)) = bounds(self.drawing.atoms.iter().map(|a| a.pos)) {
            self.pan = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
        }
    }

    /// The list the box searches: the shipped one (in a test that gives one, its own).
    fn library(&self) -> &'static Library {
        #[cfg(test)]
        if let Some(l) = self.search.library {
            return l;
        }
        Library::shipped()
    }

    /// The local results for the box's text ([`search_local`] over the shipped list and the
    /// session's cache), remade when the text changes.
    pub fn search_hits(&mut self) -> &[Hit] {
        if self.search.local.as_ref().is_none_or(|(t, _)| *t != self.search_query) {
            let cache = self.lookup.local_cache();
            let hits = search_local(&self.search_query, self.library(), &cache);
            self.search.local = Some((self.search_query.clone(), hits));
        }
        &self.search.local.as_ref().expect("just made").1
    }

    /// Whether the box offers "Search PubChem for '…'" (module doc): a query to send, no exact local
    /// hit ([`is_exact`]: however many prefix or substring hits there are, the molecule asked for may
    /// not be among them), the online setting on and PubChem reachable.
    pub fn can_search_online(&mut self) -> bool {
        let sendable = match online_query(&self.search_query) {
            Query::Empty => false,
            Query::Cid(_) => true,
            Query::Name(s) | Query::Formula(s) => s.len() <= MAX_NAME,
        };
        let q = self.search_query.clone();
        sendable && !self.search_hits().iter().any(|h| is_exact(&q, h)) && self.lookup.can_search(self.online)
    }

    /// Asks PubChem for the box's text (a later ask supersedes it: one still running stops before
    /// its next request).
    pub fn search_online(&mut self) {
        if !self.can_search_online() {
            return;
        }
        let q = online_query(&self.search_query);
        if let Some(id) = self.lookup.search(&q, self.online) {
            self.search.online = Some(Online { query: online_key(&self.search_query), id, asked: self.lookup.now(), state: OnlineState::Waiting, unanswered: true });
        }
    }

    /// The online search for the box's text as it is now (an older text's is not shown).
    fn current_online(&self) -> Option<&Online> {
        self.search.online.as_ref().filter(|o| o.query == online_key(&self.search_query))
    }

    /// The box's online search is still waiting for PubChem (the window keeps repainting).
    pub fn search_waiting(&self) -> bool {
        self.current_online().is_some_and(|o| matches!(o.state, OnlineState::Waiting))
    }

    /// A search read as not answered may still answer: the window repaints slowly for it, up to
    /// [`super::online::LATE_WAIT`] after it was asked.
    pub fn search_awaiting(&self) -> bool {
        self.current_online().is_some_and(|o| o.unanswered && !matches!(o.state, OnlineState::Waiting) && self.lookup.now().saturating_duration_since(o.asked) < super::online::LATE_WAIT)
    }

    /// Takes the online answers in: the one for the box's search replaces its state (even after it
    /// read as not answered); others are stale and dropped. One waiting past [`ONLINE_WAIT`] reads
    /// as a timeout. The online setting off, or the box's text changed, drops the search and stops
    /// it if still running: it makes no further request (Task 5 review fix 1).
    pub fn poll_search(&mut self) {
        let answers = self.lookup.poll_search();
        let now = self.lookup.now();
        let key = online_key(&self.search_query);
        if self.search.online.as_ref().is_some_and(|o| !self.online || o.query != key) && self.search.online.take().is_some_and(|o| o.unanswered) {
            self.lookup.stop_searches();
        }
        if let Some(o) = &mut self.search.online {
            for (id, a) in answers {
                if id == o.id {
                    o.unanswered = false;
                    o.state = match a {
                        Ok(v) => OnlineState::Done(v),
                        Err(e) => OnlineState::Failed(e),
                    };
                }
            }
            if matches!(o.state, OnlineState::Waiting) && now.saturating_duration_since(o.asked) >= ONLINE_WAIT {
                o.state = OnlineState::Failed(NetError::Timeout);
            }
        }
    }

    /// PubChem's answer for the box's text, when it came.
    pub fn online_hits(&self) -> Option<&[Hit]> {
        match &self.current_online()?.state {
            OnlineState::Done(v) => Some(v),
            _ => None,
        }
    }

    /// Why the box's online search got no answer, when it failed.
    pub fn online_failed(&self) -> Option<NetError> {
        match self.current_online()?.state {
            OnlineState::Failed(e) => Some(e),
            _ => None,
        }
    }

    /// The `--ui builder=<query>` screenshot step: `q` in the box, its first local result stamped
    /// (no network); the stamped molecule's name, None when nothing matched.
    pub fn search_and_stamp_first(&mut self, q: &str) -> Option<String> {
        self.search_query = q.to_string();
        let first = self.search_hits().first().and_then(|h| h.record.clone())?;
        self.stamp_record(&first);
        Some(first.name)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::online::tests::{open_gate, Canned, Gate, ManualClock, Recording};
    use super::super::online::{name_url, Lookups, Session, PUG};
    use super::*;
    use std::sync::{Arc, Condvar, Mutex, OnceLock};

    const WAIT: Duration = Duration::from_secs(30);

    fn names(hits: &[Hit]) -> Vec<&str> {
        hits.iter().map(|h| h.name.as_str()).collect()
    }

    /// A record for the search only (its graph's elements and charge; no bonds or layout).
    fn fake(name: &str, synonyms: &[&str], cid: u64, z: &[u8], charge: i8) -> Record {
        let n = z.len();
        let mut q = vec![0; n];
        q[0] = charge;
        Record {
            name: name.into(),
            synonyms: synonyms.iter().map(|s| s.to_string()).collect(),
            source: Source::PubChem { cid },
            z: z.to_vec(),
            charge: q,
            bonds: vec![],
            xyz: vec![[0.0; 3]; n],
            xy: vec![[0.0; 2]; n],
            stereo_defined: false,
        }
    }

    /// The search tests' own list (review M8: independent of what the shipped list holds), with
    /// isomers, prefixes and substrings to rank. No ethane in it (the online tests fetch it).
    pub(crate) fn test_library() -> &'static Library {
        static LIB: OnceLock<Library> = OnceLock::new();
        LIB.get_or_init(|| {
            let h = |n: usize| std::iter::repeat_n(1u8, n);
            let z = |heavy: &[u8], hs: usize| heavy.iter().copied().chain(h(hs)).collect::<Vec<u8>>();
            Library::from_records(vec![
                fake("ethanol", &["ethyl alcohol"], 702, &z(&[6, 6, 8], 6), 0),
                fake("dimethyl ether", &[], 8254, &z(&[6, 8, 6], 6), 0),
                fake("ethanolamine", &[], 700, &z(&[6, 6, 7, 8], 7), 0),
                fake("methanol", &[], 887, &z(&[6, 8], 4), 0),
                fake("methane", &[], 297, &z(&[6], 4), 0),
                fake("water", &["oxidane"], 962, &z(&[8], 2), 0),
                fake("ammonium", &["azanium"], 223, &z(&[7], 4), 1),
                fake("ammonia", &[], 222, &z(&[7], 3), 0),
                fake("hydroxide", &[], 961, &z(&[8], 1), -1),
                fake("sodium chloride", &[], 5234, &[11, 17], 0),
                fake("chlorine", &[], 24526, &[17, 17], 0),
            ])
        })
    }

    fn local(q: &str) -> Vec<Hit> {
        search_local(q, test_library(), &Cache::none())
    }

    /// A real record of the shipped list (its 2D layout and bonds: the stamping tests).
    fn seed(name: &str) -> &'static Record {
        Library::shipped().records().iter().find(|r| r.name == name).unwrap()
    }

    #[test]
    fn exact_names_rank_first() {
        let hits = local("ethanol");
        assert_eq!(names(&hits), ["ethanol", "ethanolamine", "methanol"], "exact, prefix, then containing");
        assert_eq!((hits[0].formula.as_str(), hits[0].source, hits[0].cid()), ("C2H6O", "shipped", Some(702)));
        assert_eq!(names(&local("Ethanol "))[0], "ethanol", "case and padding");
        // prefixes (the shorter first), then names containing it (the shorter first)
        assert_eq!(names(&local("eth")), ["ethanol", "ethanolamine", "methane", "methanol", "dimethyl ether"]);
        assert_eq!(names(&local("methan")), ["methane", "methanol"]);
        // by formula: both isomers, however written
        for q in ["C2H6O", "C2H5OH", "CH3OCH3", "c2h6o"] {
            assert_eq!(names(&local(q)), ["ethanol", "dimethyl ether"], "{q}");
        }
        // a synonym, a CID (leading zeros too), a charged formula (and without its charge)
        assert_eq!(names(&local("ethyl alcohol")), ["ethanol"]);
        assert_eq!(names(&local("azanium")), ["ammonium"]);
        assert_eq!(names(&local("702")), ["ethanol"]);
        assert_eq!(names(&local("000702")), ["ethanol"]);
        for q in ["NH4+", "NH₄⁺", "NH4", "nh4+"] {
            assert_eq!(names(&local(q)), ["ammonium"], "{q}");
        }
        assert!(local("NH4-").is_empty(), "the wrong charge");
        assert_eq!(names(&local("NH3")), ["ammonia"]);
        assert_eq!(names(&local("OH⁻")), ["hydroxide"]);
        // review M4: lowercase formulas split into symbols, two letters tried first
        assert_eq!(names(&local("nacl")), ["sodium chloride"]);
        assert_eq!(names(&local("cl2")), ["chlorine"]);
        assert!(local("xyzzy").is_empty());
        assert!(local("").is_empty() && local("   ").is_empty());
        // at most MAX_RESULTS, the shorter names first
        let many = Library::from_records((0..20).map(|i| fake(&format!("ethyl {}", "x".repeat(20 - i)), &[], 1000 + i as u64, &[6], 0)).collect());
        let hits = search_local("ethyl", &many, &Cache::none());
        assert_eq!(hits.len(), MAX_RESULTS);
        assert_eq!(hits[0].name, "ethyl x");
    }

    /// Review M4: the case correction.
    #[test]
    fn lowercase_formulas_split_into_symbols() {
        assert_eq!(parse_query("h2o"), Query::Formula("H2O".into()));
        assert_eq!(parse_query("cl2"), Query::Formula("Cl2".into()));
        assert_eq!(parse_query("nacl"), Query::Formula("NaCl".into()));
        assert_eq!(parse_query("NACL"), Query::Formula("NaCl".into()), "no strict reading: N, A? — split again");
        assert_eq!(parse_query("hcl"), Query::Formula("HCl".into()), "Hc is no symbol: H then Cl");
        assert_eq!(parse_query("co"), Query::Formula("CO".into()), "the builder's elements only: C and O, not cobalt");
        assert_eq!(parse_query("nh4+"), Query::Formula("NH4+".into()), "not nihonium");
        assert_eq!(parse_query("ne"), Query::Name("ne".into()), "neon, one element: a name");
        assert_eq!(parse_query("CO"), Query::Formula("CO".into()), "as written, carbon monoxide");
        assert_eq!(parse_query("ethanol"), Query::Name("ethanol".into()));
        assert_eq!(symbols("nacl").as_deref(), Some("NaCl"));
        assert_eq!(symbols("xq"), None);
        // linear however long: a run that never splits
        assert_eq!(symbols(&"jq".repeat(400)), None);
    }

    /// Review focus 5.
    #[test]
    fn odd_queries_never_panic_or_send_malformed_urls() {
        let long = "ab".repeat(150);
        let odd = ["", "   ", "000702", "H₂O", "h2o", "SO₄²⁻", "ethanol; DROP", long.as_str(), "a/b", "%2F", "ü", "\u{0}\t", "99999999999999999999999", "0", "H2Oü", "NH4ü", "C^", "H2O^+x", "Na+++++++++++", "C1000"];
        for q in odd {
            let _ = parse_query(q);
            let _ = online_query(q);
            let _ = local(q);
        }
        assert_eq!(parse_query(""), Query::Empty);
        assert_eq!(parse_query("   "), Query::Empty);
        assert_eq!(parse_query("000702"), Query::Cid(702));
        assert_eq!(parse_query("H₂O"), Query::Formula("H2O".into()));
        assert_eq!(parse_query("h2o"), Query::Formula("H2O".into()));
        assert_eq!(parse_query("ethanol; DROP"), Query::Name("ethanol; DROP".into()));
        assert_eq!(parse_query("  ethyl   alcohol "), Query::Name("ethyl alcohol".into()));
        assert_eq!(parse_query("He"), Query::Name("He".into()), "one element alone reads as a name");
        assert_eq!(parse_query("SO₄²⁻"), Query::Formula("SO4^2-".into()), "a superscript charge stays a charge");
        assert_eq!(parse_query("H2Oü"), Query::Name("H2Oü".into()));
        // review M3: a formula goes online as typed (trimmed), never in the internal notation
        assert_eq!(online_query(" SO₄²⁻ "), Query::Formula("SO42-".into()));
        assert_eq!(online_query("H₂O"), Query::Formula("H2O".into()));
        assert_eq!(online_query("h2o"), Query::Formula("h2o".into()), "the user's case kept");
        assert_eq!(online_query("NH\u{2084}\u{207A}\u{0}"), Query::Formula("NH4+".into()), "a control character dropped");
        assert_eq!(online_query("ethyl  alcohol"), Query::Name("ethyl alcohol".into()));
        assert_eq!(online_query("SO4^2-"), Query::Formula("SO42-".into()), "a typed caret is not sent");
        for q in ["H₂O", "h2o", "H2O", "OH2"] {
            assert_eq!(names(&local(q)), ["water"], "{q}");
        }
        // every URL the online search builds: ASCII, percent-encoded, under PubChem's root (a
        // recording transport over canned answers: never the network)
        let urls = Arc::new(Mutex::new(Vec::new()));
        let l = Lookups::with_clock(Box::new(Recording { inner: Canned::new(Default::default()), urls: urls.clone() }), Cache::none(), ManualClock::new());
        for q in odd {
            let id = l.search(&online_query(q));
            let (got, hits) = l.wait_search(WAIT).expect("an answer");
            assert_eq!(got, id);
            assert_eq!(hits, Ok(vec![]), "{q:?}: canned 404s are no results, never an error");
        }
        let urls = urls.lock().unwrap().clone();
        assert!(!urls.is_empty());
        for u in &urls {
            assert!(u.is_ascii() && u.starts_with(PUG), "{u}");
            let path = &u[PUG.len()..];
            assert!(path.bytes().all(|b| b.is_ascii_alphanumeric() || b"-._~%/?=".contains(&b)), "{u}");
            for (i, _) in path.match_indices('%') {
                assert!(path.len() >= i + 3 && path[i + 1..i + 3].bytes().all(|b| b.is_ascii_hexdigit()), "{u}");
            }
            // one name or CID segment: an encoded '/' never splits the path
            let segs: Vec<&str> = path.split('?').next().unwrap().split('/').collect();
            assert!(segs.len() == 4 || segs.len() == 5, "{u}");
            assert!(!u.contains("SO4%5E"), "no ^ notation sent for a superscript: {u}");
        }
        assert!(urls.iter().any(|u| u.contains("compound/cid/702/")), "{urls:?}");
        assert!(urls.contains(&name_url("a/b")) && name_url("a/b").contains("compound/name/a%2Fb/cids/TXT"), "{urls:?}");
        assert!(urls.contains(&name_url("h2o")) && urls.contains(&name_url("H2O")) && urls.contains(&name_url("SO42-")), "formulas in plain ASCII: {urls:?}");
        assert!(!urls.iter().any(|u| u.contains("%E2%82")), "no Unicode subscript sent: {urls:?}");
        assert!(!urls.iter().any(|u| u.contains(&long)), "a name past MAX_NAME is not sent");
    }

    /// Review M2: the combined list is at most MAX_RESULTS; PubChem's rows fill what the local ones
    /// leave, duplicates (by CID) left out, and the rest is counted.
    #[test]
    fn the_combined_list_is_capped() {
        let lib = test_library().records();
        let local: Vec<Hit> = lib[..6].iter().map(|r| Hit::of(r, "shipped")).collect();
        let online: Vec<Hit> = std::iter::once(Hit::of(&lib[0], "PubChem")).chain((0..4).map(|i| Hit::of(&fake("x", &[], 90 + i, &[6], 0), "PubChem"))).collect();
        // 6 local, 4 fresh from PubChem: the 2 lowest local rows give way
        let (kept, rows) = combined_rows(&local, &online);
        assert_eq!(rows.iter().map(|h| h.cid()).collect::<Vec<_>>(), [Some(90), Some(91), Some(92), Some(93)]);
        assert_eq!(names(&kept.into_iter().cloned().collect::<Vec<_>>()), names(&local[..4]), "the best-ranked local rows stay");
        let (kept, rows) = combined_rows(&local[..1], &online[..1]);
        assert_eq!((kept.len(), rows.len()), (1, 0), "only a duplicate");
        // fresh PubChem rows past MAX_RESULTS: the list is PubChem's alone, still at most 8
        let flood: Vec<Hit> = (0..10).map(|i| Hit::of(&fake("y", &[], 200 + i, &[6], 0), "PubChem")).collect();
        let (kept, rows) = combined_rows(&local, &flood);
        assert_eq!((kept.len(), rows.len()), (0, MAX_RESULTS));
    }

    /// Fix round 2: a large list whose names all contain the query, none exactly, still offers
    /// PubChem (the molecule asked for may not be among them); an exact local hit does not.
    #[test]
    fn a_crowded_query_without_an_exact_hit_still_offers_pubchem() {
        let many = Box::leak(Box::new(Library::from_records((0..9).map(|i| fake(&format!("{i}-ethyl-x"), &[], 1000 + i, &[6], 0)).collect())));
        let clock = ManualClock::new();
        let mut b = Builder { search_query: "ethyl".into(), ..Default::default() };
        b.search.library = Some(many);
        b.lookup = Session::with(Lookups::with_clock(Box::new(Canned::new(Default::default())), Cache::none(), clock.clone()), Cache::none(), clock);
        assert_eq!(b.search_hits().len(), MAX_RESULTS, "a full local list");
        assert!(b.can_search_online(), "no exact hit: PubChem is offered");
        b.search_query = "3-ethyl-x".into();
        assert!(!b.can_search_online(), "an exact name: nothing to ask");
        b.search_query = "1003".into();
        assert!(!b.can_search_online(), "an exact CID: nothing to ask");
        b.search_query = "1003 ".into();
        assert!(!b.can_search_online(), "padding changes nothing");
    }

    /// A builder searching [`test_library`] through a session whose PubChem is `t`, the window's
    /// clock apart from the worker's (the worker's request deadlines do not run out while the
    /// test moves the window's clock).
    fn builder_over(t: Box<dyn super::super::online::Transport>, query: &str) -> (Builder, Arc<ManualClock>) {
        let clock = ManualClock::new();
        let mut b = Builder { search_query: query.into(), ..Default::default() };
        b.search.library = Some(test_library());
        b.lookup = Session::with(Lookups::with_clock(t, Cache::none(), ManualClock::new()), Cache::none(), clock.clone());
        (b, clock)
    }

    fn ethane_by_name() -> std::collections::HashMap<String, Result<String, NetError>> {
        let mut answers = super::super::online::tests::ethane_answers("CC");
        answers.insert(name_url("ethane"), Ok("6324\n".into()));
        answers
    }

    /// Blocks until the transport saw `n` requests (the worker is then at the gate).
    fn until(urls: &Arc<Mutex<Vec<String>>>, n: usize) {
        let t0 = std::time::Instant::now();
        while urls.lock().unwrap().len() < n {
            assert!(t0.elapsed() < WAIT, "the worker never asked");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Review fix 1: the online setting going off stops a search in flight — no request after it —
    /// and a newer search stops an older one the same way, so it does not wait behind it.
    #[test]
    fn turning_the_setting_off_stops_a_search_in_flight() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let urls = Arc::new(Mutex::new(Vec::new()));
        let t = Recording { inner: Gate { inner: Canned::new(ethane_by_name()), open: gate.clone() }, urls: urls.clone() };
        let (mut b, _) = builder_over(Box::new(t), "ethane");
        b.search_online();
        until(&urls, 1); // the name lookup is out, held at the gate
        b.online = false;
        b.poll_search();
        assert!(!b.search_waiting() && b.online_hits().is_none());
        open_gate(&gate);
        // the stopped search's answer (offline: it stopped before its next request) is dropped
        assert!(b.lookup.wait_search(WAIT));
        assert_eq!(*urls.lock().unwrap(), [name_url("ethane")], "no request after the setting went off");
        b.poll_search();
        assert!(b.online_hits().is_none() && b.online_failed().is_none());
        // on again: a search goes out in full
        b.online = true;
        b.search_online();
        assert!(b.lookup.wait_search(WAIT));
        b.poll_search();
        assert_eq!(b.online_hits().map(|h| h.len()), Some(1));

        // a newer search stops the one in flight before its next request
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let urls = Arc::new(Mutex::new(Vec::new()));
        let t = Recording { inner: Gate { inner: Canned::new(ethane_by_name()), open: gate.clone() }, urls: urls.clone() };
        let (mut b, _) = builder_over(Box::new(t), "ethane");
        b.search_online();
        until(&urls, 1);
        b.search_query = "propane".into();
        b.search_online();
        open_gate(&gate);
        until(&urls, 2);
        assert!(b.lookup.wait_search(WAIT) && b.lookup.wait_search(WAIT), "both answer");
        assert_eq!(*urls.lock().unwrap(), [name_url("ethane"), name_url("propane")], "ethane's 3D record was never asked");
        b.poll_search();
        assert_eq!(b.online_hits(), Some(&[][..]), "propane: no results");
    }

    /// Review M1: a search read as not answered after ONLINE_WAIT keeps the window repainting
    /// slowly (as the drawn lookups' LATE_WAIT), so its late answer shows without input.
    #[test]
    fn a_late_search_answer_keeps_the_window_repainting() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (mut b, clock) = builder_over(Box::new(Gate { inner: Canned::new(ethane_by_name()), open: gate.clone() }), "ethane");
        b.search_online();
        assert!(b.search_waiting() && !b.search_awaiting());
        clock.advance(ONLINE_WAIT);
        b.poll_search();
        assert_eq!(b.online_failed(), Some(NetError::Timeout));
        assert!(!b.search_waiting() && b.search_awaiting(), "repaint, slowly, for the late answer");
        open_gate(&gate);
        assert!(b.lookup.wait_search(WAIT));
        b.poll_search();
        assert_eq!(b.online_hits().map(|h| h[0].name.as_str()), Some("Ethane"));
        assert!(!b.search_awaiting());
        // one that never answers: the repaint ends LATE_WAIT after the ask
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (mut b, clock) = builder_over(Box::new(Gate { inner: Canned::new(Default::default()), open: gate.clone() }), "propane");
        b.search_online();
        clock.advance(ONLINE_WAIT);
        b.poll_search();
        assert!(b.search_awaiting());
        clock.advance(super::super::online::LATE_WAIT);
        assert!(!b.search_awaiting());
        open_gate(&gate);
    }

    #[test]
    fn stamping_is_one_undo_step_beside_the_drawing() {
        let mut b = Builder::default();
        super::super::shape::tests::draw_record(&mut b.drawing, seed("water"), [0.0, 0.0], &[0, 1, 2]);
        let water = b.drawing.clone();
        let undo_depth = b.undo.len();
        b.stamp_record(seed("benzene"));
        assert_eq!(b.drawing.atoms.len(), water.atoms.len() + 12);
        assert_eq!(b.drawing.bonds.len(), water.bonds.len() + 12);
        assert_eq!(b.undo.len(), undo_depth + 1, "one checkpoint");
        let max_x = water.atoms.iter().map(|a| a.pos[0]).fold(f32::MIN, f32::max);
        for a in &b.drawing.atoms[water.atoms.len()..] {
            assert!(a.pos[0] > max_x, "{:?} not right of {max_x}", a.pos);
        }
        let min_stamped = b.drawing.atoms[water.atoms.len()..].iter().map(|a| a.pos[0]).fold(f32::MAX, f32::min);
        assert!((min_stamped - max_x - 1.5).abs() < 1e-5, "1.5 units apart: {min_stamped} vs {max_x}");
        // the graph copied: elements, charges, bond orders (Kekulé)
        let r = seed("benzene");
        let n = water.atoms.len();
        assert_eq!(b.drawing.atoms[n..].iter().map(|a| (a.z, a.charge)).collect::<Vec<_>>(), r.z.iter().zip(&r.charge).map(|(&z, &q)| (z, q)).collect::<Vec<_>>());
        assert_eq!(b.drawing.bonds[water.bonds.len()..].iter().map(|k| (k.a - n, k.b - n, k.order)).collect::<Vec<_>>(), r.bonds);
        assert_eq!(b.selection.len(), 12, "the stamp is selected, to move it as one");
        b.undo();
        assert_eq!(b.drawing, water, "one undo restores the water exactly");
        b.redo();
        assert_eq!(b.drawing.atoms.len(), 15);
        // a charged record keeps its charges; an empty canvas centres the stamp on the origin
        let mut e = Builder::default();
        e.stamp_record(seed("ammonium"));
        assert_eq!(e.drawing.charge(), 1);
        let (lo, hi) = bounds(e.drawing.atoms.iter().map(|a| a.pos));
        assert!(((lo[0] + hi[0]) / 2.0).abs() < 1e-5 && ((lo[1] + hi[1]) / 2.0).abs() < 1e-5, "{lo:?} {hi:?}");
        e.undo();
        assert!(e.drawing.atoms.is_empty());
    }

    fn bounds(p: impl Iterator<Item = [f32; 2]>) -> ([f32; 2], [f32; 2]) {
        p.fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), q| ([lo[0].min(q[0]), lo[1].min(q[1])], [hi[0].max(q[0]), hi[1].max(q[1])]))
    }

    #[test]
    fn the_stamped_layout_has_unit_bonds() {
        let mut b = Builder::default();
        super::super::shape::tests::draw_record(&mut b.drawing, seed("water"), [0.0, 0.0], &[0, 1, 2]);
        let n = b.drawing.atoms.len();
        let k = b.drawing.bonds.len();
        b.stamp_record(seed("benzene"));
        let d = &b.drawing;
        let len = |a: usize, c: usize| ((d.atoms[a].pos[0] - d.atoms[c].pos[0]) as f64).hypot((d.atoms[a].pos[1] - d.atoms[c].pos[1]) as f64);
        let stamped = &d.bonds[k..];
        assert!(stamped.iter().all(|bd| bd.a >= n && bd.b >= n));
        let mean = stamped.iter().map(|bd| len(bd.a, bd.b)).sum::<f64>() / stamped.len() as f64;
        assert!((mean - 1.0).abs() < 1e-6, "mean drawn bond {mean}");
        // and the stamped drawing is benzene to the lookup: the database shape on hand-off
        let frag: Vec<usize> = (n..d.atoms.len()).collect();
        assert!(matches!(super::super::iso::map_fragment(d, &frag, seed("benzene")), super::super::iso::MapResult::Mapped(_)));
    }

    /// The search box's flow through the window's session: local hits for every keystroke, the
    /// online row only with the setting on and PubChem reachable, its answer (canned) listed with
    /// the local ones' duplicates left out, and a query typed since makes a late answer stale.
    #[test]
    fn the_online_search_runs_through_the_session() {
        use super::super::online::tests::{ethane_answers, temp_dir};
        let dir = temp_dir("search-session");
        let clock = ManualClock::new();
        let cache = Cache::open(dir.clone());
        let mut answers = ethane_answers("CC");
        answers.insert(name_url("ethane"), Ok("6324\n".into()));
        let canned = Canned::new(answers);
        // a test build: no network, so the app's own session offers no online row
        let mut b = Builder { search_query: "ethane".into(), ..Default::default() };
        b.search.library = Some(test_library());
        assert!(!b.can_search_online());
        b.lookup = Session::with(Lookups::with_clock(Box::new(canned.clone()), cache.clone(), clock.clone()), cache, clock.clone());
        assert!(b.can_search_online());
        b.online = false;
        assert!(!b.can_search_online(), "the setting off: no row");
        b.online = true;
        assert_eq!(names(b.search_hits()), ["methane"], "not in the list (methane contains it)");
        b.search_online();
        assert!(b.search_waiting());
        assert!(b.lookup.wait_search(WAIT));
        b.poll_search();
        assert!(!b.search_waiting());
        let hits = b.online_hits().expect("answered").to_vec();
        assert_eq!((hits.len(), hits[0].name.as_str(), hits[0].source), (1, "Ethane", "PubChem"));
        // kept: the same query (in any case) now finds it in the cache too
        b.search_query = "Ethane".into();
        let local = b.search_hits().to_vec();
        assert_eq!((local.len(), local[0].name.as_str(), local[0].source), (2, "Ethane", "cached"));
        assert_eq!(local[0].cid(), Some(6324));
        assert!(b.online_hits().is_some(), "the same query");
        b.search_query = "ethan".into();
        assert!(b.online_hits().is_none(), "the online answer belonged to the query before");
        let prefix = b.search_hits().to_vec();
        assert_eq!(names(&prefix)[..2], ["Ethane", "ethanol"], "the shorter name first");
        b.stamp_record(local[0].record.as_ref().unwrap());
        assert_eq!(b.drawing.atoms.len(), 8);
        // a failed search says so and never panics
        let mut down = Canned::new(Default::default());
        down.otherwise = Err(NetError::Status(503));
        b.lookup = Session::with(Lookups::with_clock(Box::new(down), Cache::none(), clock.clone()), Cache::none(), clock.clone());
        b.search_query = "propane".into();
        b.search_online();
        assert!(b.lookup.wait_search(WAIT));
        b.poll_search();
        assert_eq!(b.online_failed(), Some(NetError::Status(503)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `--ui builder=<query>` screenshot key's step: the local search alone, the first hit
    /// stamped.
    #[test]
    fn search_and_stamp_first_stamps_the_top_local_hit() {
        let mut b = Builder::default();
        b.search.library = Some(Box::leak(Box::new(Library::from_records(vec![seed("ethanol").clone(), seed("methane").clone()]))));
        assert_eq!(b.search_and_stamp_first("ethanol").as_deref(), Some("ethanol"));
        assert_eq!(b.search_query, "ethanol");
        assert_eq!(b.drawing.atoms.len(), 9);
        assert_eq!(b.search_and_stamp_first("xyzzy"), None);
        assert_eq!(b.drawing.atoms.len(), 9);
    }
}
