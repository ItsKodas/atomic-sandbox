//! The builder's search box (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md`
//! §4 Goal 2, §7, §8): a name, a formula or a PubChem CID typed at the top of the palette column
//! lists the matching molecules; picking one stamps it onto the canvas as a tidy 2D drawing (the
//! record's own layout, every hydrogen explicit) beside what is drawn, as one undo step.
//!
//! **The query** ([`parse_query`]) is compared after [`fold`]: Unicode subscripts read as digits
//! (`H₂O` is `H2O`), superscript charges as `^` and the charge (`OH⁻` is `OH^-`), whitespace runs
//! as one space. All digits is a CID (`000702` is 702); a formula written as chemists write it
//! (`C2H5OH`, `NH4+`, `h2o` with its letters uppercased when that is the only reading) is a
//! formula; anything else is a name.
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
//! click (or Enter with no local result) asks it. A name or a formula goes to PubChem's name
//! endpoint, a CID to its compound; the answer is listed below the local results (duplicates by
//! CID left out), and a failure is one short line, never a toast. A name holding `/` is sent
//! percent-encoded (`%2F`) in the path segment, which PubChem often refuses: that is a 4xx, read as
//! "no results on PubChem" (PUG REST's POST form would carry it, but the transport is GET-only).
//!
//! **Rule 1 (spec §3).** A record only places atoms on the canvas: it never leaves the builder.
use super::online::{Cache, NetError};
use super::records::{hill, Library, Record, Source};
use super::{Atom, Bond, Builder};
use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};
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
            c => (c, false),
        };
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        if is_sup && !sup {
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

/// [`formula`], else (an ASCII query with a count somewhere) the same with its letters uppercased,
/// which is the only reading of `h2o` or `nh4+`: the formula's text (case corrected) and its reading.
fn formula_any_case(s: &str) -> Option<(String, Formula)> {
    if let Some(f) = formula(s) {
        return Some((s.to_string(), f));
    }
    if !s.is_ascii() || !s.bytes().any(|b| b.is_ascii_digit()) {
        return None;
    }
    let up = s.to_ascii_uppercase();
    formula(&up).map(|f| (up, f))
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

/// The PubChem search asked for one query: its id, when, and how it went.
pub(crate) struct Online {
    /// The query's [`fold`], lowercased: the box's text it belongs to.
    query: String,
    id: u64,
    asked: Instant,
    state: OnlineState,
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

    /// The local results for the box's text ([`search_local`] over the shipped list and the
    /// session's cache), remade when the text changes.
    pub fn search_hits(&mut self) -> &[Hit] {
        if self.search.local.as_ref().is_none_or(|(t, _)| *t != self.search_query) {
            let cache = self.lookup.local_cache();
            let hits = search_local(&self.search_query, Library::shipped(), &cache);
            self.search.local = Some((self.search_query.clone(), hits));
        }
        &self.search.local.as_ref().expect("just made").1
    }

    /// Whether the box offers "Search PubChem for '…'" (module doc): a query to send, the online
    /// setting on and PubChem reachable.
    pub fn can_search_online(&mut self) -> bool {
        let sendable = match parse_query(&self.search_query) {
            Query::Empty => false,
            Query::Cid(_) => true,
            Query::Name(s) | Query::Formula(s) => s.len() <= MAX_NAME,
        };
        sendable && self.lookup.can_search(self.online)
    }

    /// Asks PubChem for the box's text (a later ask supersedes it).
    pub fn search_online(&mut self) {
        if !self.can_search_online() {
            return;
        }
        let q = parse_query(&self.search_query);
        if let Some(id) = self.lookup.search(&q, self.online) {
            self.search.online = Some(Online { query: online_key(&self.search_query), id, asked: self.lookup.now(), state: OnlineState::Waiting });
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

    /// Takes the online answers in: the one for the box's search replaces its state (even after it
    /// read as not answered); others are stale and dropped. One waiting past [`ONLINE_WAIT`] reads
    /// as a timeout.
    pub fn poll_search(&mut self) {
        let answers = self.lookup.poll_search();
        let now = self.lookup.now();
        if let Some(o) = &mut self.search.online {
            for (id, a) in answers {
                if id == o.id {
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
mod tests {
    use super::super::online::tests::{Canned, ManualClock, Recording};
    use super::super::online::{Lookups, PUG};
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(30);

    fn names(hits: &[Hit]) -> Vec<&str> {
        hits.iter().map(|h| h.name.as_str()).collect()
    }

    fn local(q: &str) -> Vec<Hit> {
        search_local(q, Library::shipped(), &Cache::none())
    }

    fn seed(name: &str) -> &'static Record {
        Library::shipped().records().iter().find(|r| r.name == name).unwrap()
    }

    #[test]
    fn exact_names_rank_first() {
        let hits = local("ethanol");
        assert_eq!(hits.first().map(|h| h.name.as_str()), Some("ethanol"), "{:?}", names(&hits));
        assert_eq!((hits[0].formula.as_str(), hits[0].source), ("C2H6O", "shipped"));
        assert_eq!(hits[0].record.as_ref(), Some(seed("ethanol")));
        assert!(names(&local("eth")).contains(&"ethanol"), "{:?}", names(&local("eth")));
        assert!(names(&local("Ethanol ")).first() == Some(&"ethanol"), "case and padding");
        assert_eq!(names(&local("C2H6O")), vec!["ethanol"], "by formula");
        assert_eq!(names(&local("C2H5OH")), vec!["ethanol"], "a formula as chemists write it");
        // a synonym, a CID, a charged formula (and without its charge)
        assert_eq!(names(&local("ethyl alcohol")), vec!["ethanol"]);
        assert_eq!(names(&local("702")), vec!["ethanol"]);
        assert_eq!(names(&local("NH4+")), vec!["ammonium"]);
        assert_eq!(names(&local("OH⁻")), vec!["hydroxide"]);
        assert_eq!(names(&local("NH4-")), Vec::<&str>::new(), "the wrong charge");
        assert!(names(&local("NH4")).contains(&"ammonium"));
        // a prefix of another name ranks below the exact one: "water" itself first
        assert_eq!(names(&local("water"))[0], "water");
        assert!(local("xyzzy").is_empty());
        assert!(local("").is_empty() && local("   ").is_empty());
        assert!(local("e").len() <= MAX_RESULTS);
    }

    /// Review focus 5.
    #[test]
    fn odd_queries_never_panic_or_send_malformed_urls() {
        let long = "ab".repeat(150);
        let odd = ["", "   ", "000702", "H₂O", "h2o", "ethanol; DROP", long.as_str(), "a/b", "%2F", "ü", "\u{0}\t", "99999999999999999999999", "0", "H2Oü", "NH4ü", "C^", "H2O^+x", "Na+++++++++++", "C1000"];
        for q in odd {
            let _ = parse_query(q);
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
        assert_eq!(parse_query("CO"), Query::Formula("CO".into()));
        assert_eq!(parse_query("SO₄²⁻"), Query::Formula("SO4^2-".into()), "a superscript charge stays a charge");
        assert_eq!(parse_query("H2Oü"), Query::Name("H2Oü".into()));
        assert_eq!(names(&local("NH₄⁺")), vec!["ammonium"]);
        for q in ["H₂O", "h2o", "H2O", "OH2"] {
            assert_eq!(names(&local(q)), vec!["water"], "{q}");
        }
        // every URL the online search builds: ASCII, percent-encoded, under PubChem's root (a
        // recording transport over canned answers: never the network)
        let urls = Arc::new(Mutex::new(Vec::new()));
        let l = Lookups::with_clock(Box::new(Recording { inner: Canned::new(Default::default()), urls: urls.clone() }), Cache::none(), ManualClock::new());
        for q in odd {
            let id = l.search(&parse_query(q));
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
        }
        assert!(urls.iter().any(|u| u.contains("compound/cid/702/")), "{urls:?}");
        assert!(urls.iter().any(|u| u.contains("compound/name/a%2Fb/cids/TXT")), "{urls:?}");
        assert!(!urls.iter().any(|u| u.contains(&long)), "a name past MAX_NAME is not sent");
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
        use super::super::online::{name_url, Session};
        let dir = temp_dir("search-session");
        let clock = ManualClock::new();
        let cache = Cache::open(dir.clone());
        let mut answers = ethane_answers("CC");
        answers.insert(name_url("ethane"), Ok("6324\n".into()));
        let canned = Canned::new(answers);
        // a test build: no network, so the app's own session offers no online row
        let mut b = Builder { search_query: "ethane".into(), ..Default::default() };
        assert!(!b.can_search_online());
        b.lookup = Session::with(Lookups::with_clock(Box::new(canned.clone()), cache.clone(), clock.clone()), cache, clock.clone());
        assert!(b.can_search_online());
        b.online = false;
        assert!(!b.can_search_online(), "the setting off: no row");
        b.online = true;
        assert_eq!(names(b.search_hits()), ["methane"], "not in the shipped list (methane contains it)");
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
        assert_eq!(b.search_and_stamp_first("ethanol").as_deref(), Some("ethanol"));
        assert_eq!(b.search_query, "ethanol");
        assert_eq!(b.drawing.atoms.len(), 9);
        assert_eq!(b.search_and_stamp_first("xyzzy"), None);
        assert_eq!(b.drawing.atoms.len(), 9);
    }
}
