//! The online lookup (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md` §6.3–6.4,
//! §7): a drawn fragment that neither the shipped list nor the molecule cache knows is looked up on
//! NCBI's PubChem, off the UI thread, and a record found there is kept in the cache.
//!
//! **The calls (PUG REST, HTTPS, [`PUG`]).** A fragment is written as a SMILES ([`smiles`]) and
//! resolved to compound IDs; for each CID in turn (at most [`MAX_CIDS`]) its 3D SDF is fetched and
//! mapped onto the drawn fragment with the same exact rule as the shipped list
//! ([`iso::map_fragment`]). The first CID that maps is the answer; its 2D SDF (the canvas layout),
//! its title (the label's name) and its isomeric SMILES (stereo marks `@`, `/`, `\` mean the record
//! is one stereoisomer, [`Record::stereo_defined`]) complete the record. A 404 on the 3D SDF means
//! "not in the database". Only the structure string, a name or a CID ever goes into a URL,
//! percent-encoded ([`encode`]); the user agent names the builder and its version only
//! ([`user_agent`]).
//!
//! **Never blocking, never stale.** [`Lookups`] owns one worker thread; [`Session`] (one per
//! builder) asks it once the drawing's graph has been unchanged for [`DEBOUNCE`], keeps at most
//! [`MIN_INTERVAL`] between requests (PubChem allows 5 a second), and gives each lookup [`TIMEOUT`]
//! (the worker's requests share that deadline, and the session marks a lookup still unanswered then
//! offline on its own, so the window never waits longer). Every answer carries the drawing
//! generation it was asked for; the generation goes up on every change of the drawing's graph, and
//! an answer for an older one is dropped on arrival — it never labels or places a newer drawing.
//!
//! **No network** ([`network_allowed`]): off under `--replay` and `--screenshot` (`main` calls
//! [`set_network_allowed`]) and by default in a test build; the Builder's setting (`Builder::online`)
//! turns it off per user. Either way no request is made and unknown fragments read "approximate
//! (offline)". The real transport ([`UreqTransport`]) checks the switch itself too, and tests use
//! canned transports that never reach the network.
//!
//! **The cache** ([`Cache`], `%LOCALAPPDATA%\atomic-sandbox\molecules\`): one file per CID in the
//! shipped list's record format ([`records::write`]), named `<Hill formula>_<charge>_<CID>.txt` so
//! that the folder listing is the index by formula and charge. A file that does not parse, does not
//! validate or disagrees with its name is skipped (and so refetched), never a crash.
//!
//! **Rule 1 (spec §3).** As for the shipped list: a fetched record only places nuclei in the
//! builder.
use super::iso::{self, MapResult};
use super::records::{self, canvas_layout, hill, parse_sdf, Library, Record, Source, MAX_ATOMS};
use super::search::{Hit, Query, MAX_NAME};
use super::shape::{LocalOnly, Lookup, RecordSource};
use super::{Atom, Bond, Drawing};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use units::elements::symbol;

/// How long the drawing's graph must stay unchanged before its unknown fragments are looked up.
pub const DEBOUNCE: Duration = Duration::from_secs(1);
/// The most one lookup may take, all its requests together (spec §7).
pub const TIMEOUT: Duration = Duration::from_secs(5);
/// The least time between two requests (PubChem's published limit is 5 a second).
pub const MIN_INTERVAL: Duration = Duration::from_millis(200);
/// How long after a lookup was asked the window still repaints for its late answer
/// ([`Session::awaiting`]): several fragments' lookups queued one behind another, each within
/// [`TIMEOUT`].
pub const LATE_WAIT: Duration = Duration::from_secs(60);
/// PubChem's PUG REST root: every request is under it.
pub const PUG: &str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/";
/// The most CIDs one structure lookup tries.
pub const MAX_CIDS: usize = 5;
/// The most CIDs one online search turns into results.
pub const MAX_SEARCH: usize = 3;

// ── the process-wide switch ──

static NETWORK: AtomicBool = AtomicBool::new(!cfg!(test));

/// Allows or forbids the builder's network for the whole process: `main` forbids it under
/// `--replay` and `--screenshot`; a test build starts forbidden.
pub fn set_network_allowed(on: bool) {
    NETWORK.store(on, Ordering::SeqCst);
}

/// Whether the builder may reach the network at all.
pub fn network_allowed() -> bool {
    NETWORK.load(Ordering::SeqCst)
}

// ── the SMILES writer ──

/// The fragment as a SMILES (Kekulé, no aromatic atoms): every atom in brackets with its hydrogen
/// count and formal charge (`[CH3]`, `[NH4+]`, `[O-]`), so nothing is implied by valence and a
/// radical stays a radical. A neutral hydrogen with one single bond to a non-hydrogen is counted
/// on that atom; every other hydrogen is an atom of its own (`[H][H]`, `[H+]`). Bonds of order 2
/// and 3 are `=` and `#`; rings close with digits (`%nn` past 9), reused once closed. The walk is
/// depth-first from the lowest atom, neighbours in index order, so the string is deterministic.
pub fn smiles(d: &Drawing, frag: &[usize]) -> String {
    let n = frag.len();
    let local: HashMap<usize, usize> = frag.iter().enumerate().map(|(i, &a)| (a, i)).collect();
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for b in &d.bonds {
        if let (Some(&p), Some(&q)) = (local.get(&b.a), local.get(&b.b)) {
            adj[p].push((q, b.order));
            adj[q].push((p, b.order));
        }
    }
    for a in &mut adj {
        a.sort_unstable();
    }
    let atom = |i: usize| d.atoms[frag[i]];
    let folded: Vec<bool> = (0..n).map(|i| atom(i).z == 1 && atom(i).charge == 0 && adj[i].len() == 1 && adj[i][0].1 == 1 && atom(adj[i][0].0).z != 1).collect();
    let hcount: Vec<usize> = (0..n).map(|i| adj[i].iter().filter(|&&(j, _)| folded[j]).count()).collect();
    let mut walk = Walk { adj: &adj, folded: &folded, rank: vec![usize::MAX; n], next: 0, children: vec![Vec::new(); n], rings: Vec::new() };
    let mut roots = Vec::new();
    for i in 0..n {
        if !folded[i] && walk.rank[i] == usize::MAX {
            roots.push(i);
            walk.visit(i, None);
        }
    }
    let mut out = Writer { walk: &walk, text: String::new(), in_use: Vec::new(), digit: HashMap::new(), label: &|i| atom_text(atom(i).z, hcount[i], atom(i).charge) };
    for (k, &r) in roots.iter().enumerate() {
        if k > 0 {
            out.text.push('.');
        }
        out.write(r);
    }
    out.text
}

/// `[Sym]`, `[SymH]`, `[SymHn]` with the charge (`+`, `-`, `+2`, …).
fn atom_text(z: u8, h: usize, q: i8) -> String {
    let mut t = format!("[{}", symbol(z));
    if h > 0 {
        t.push('H');
        if h > 1 {
            t += &h.to_string();
        }
    }
    match q {
        0 => {}
        1 => t.push('+'),
        -1 => t.push('-'),
        q if q > 0 => t += &format!("+{q}"),
        q => t += &format!("-{}", -(q as i32)),
    }
    t.push(']');
    t
}

fn bond_symbol(order: u8) -> &'static str {
    match order {
        2 => "=",
        3 => "#",
        _ => "",
    }
}

/// The depth-first spanning forest: each atom's tree children and the ring bonds (opened at the
/// atom reached first).
struct Walk<'a> {
    adj: &'a [Vec<(usize, u8)>],
    folded: &'a [bool],
    rank: Vec<usize>,
    next: usize,
    children: Vec<Vec<(usize, u8)>>,
    /// (opened at, closed at, order).
    rings: Vec<(usize, usize, u8)>,
}

impl Walk<'_> {
    fn visit(&mut self, u: usize, parent: Option<usize>) {
        self.rank[u] = self.next;
        self.next += 1;
        let adj = self.adj;
        for &(v, o) in &adj[u] {
            if self.folded[v] || Some(v) == parent {
                continue;
            }
            if self.rank[v] == usize::MAX {
                self.children[u].push((v, o));
                self.visit(v, Some(u));
            } else if self.rank[v] < self.rank[u] {
                self.rings.push((v, u, o));
            }
        }
    }
}

struct Writer<'a> {
    walk: &'a Walk<'a>,
    text: String,
    /// Ring digits open now (index = digit).
    in_use: Vec<bool>,
    /// The digit of each open ring bond.
    digit: HashMap<(usize, usize), usize>,
    label: &'a dyn Fn(usize) -> String,
}

impl Writer<'_> {
    fn write(&mut self, u: usize) {
        self.text += &(self.label)(u);
        let rings = &self.walk.rings;
        // the closings' digits stay taken until this atom's openings have theirs
        let closing: Vec<usize> = rings.iter().filter(|r| r.1 == u).map(|r| self.digit.remove(&(r.0, r.1)).expect("a ring bond opened earlier")).collect();
        for &k in &closing {
            self.text += &digit_text(k);
        }
        for r in rings.iter().filter(|r| r.0 == u) {
            let k = (1..).find(|&k| !self.in_use.get(k).copied().unwrap_or(false)).unwrap();
            if self.in_use.len() <= k {
                self.in_use.resize(k + 1, false);
            }
            self.in_use[k] = true;
            self.digit.insert((r.0, r.1), k);
            self.text += bond_symbol(r.2);
            self.text += &digit_text(k);
        }
        for k in closing {
            self.in_use[k] = false;
        }
        let children = &self.walk.children[u];
        for (i, &(v, o)) in children.iter().enumerate() {
            let last = i + 1 == children.len();
            if !last {
                self.text.push('(');
            }
            self.text += bond_symbol(o);
            self.write(v);
            if !last {
                self.text.push(')');
            }
        }
    }
}

fn digit_text(k: usize) -> String {
    if k < 10 {
        k.to_string()
    } else {
        format!("%{k:02}")
    }
}

// ── URLs ──

/// Percent-encodes everything but RFC 3986's unreserved characters, byte by byte (UTF-8).
pub fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out += &format!("%{b:02X}");
        }
    }
    out
}

/// A structure's CIDs: `…/compound/smiles/<smiles>/cids/TXT`.
pub fn smiles_url(s: &str) -> String {
    format!("{PUG}compound/smiles/{}/cids/TXT", encode(s))
}

/// A compound's SDF, `dim` "3d" or "2d": `…/compound/cid/<cid>/SDF?record_type=<dim>`.
pub fn sdf_url(cid: u64, dim: &str) -> String {
    format!("{PUG}compound/cid/{cid}/SDF?record_type={dim}")
}

/// A compound's title: `…/compound/cid/<cid>/property/Title/TXT`.
pub fn title_url(cid: u64) -> String {
    format!("{PUG}compound/cid/{cid}/property/Title/TXT")
}

/// A compound's isomeric SMILES (its stereo marks): `…/compound/cid/<cid>/property/IsomericSMILES/TXT`.
pub fn stereo_url(cid: u64) -> String {
    format!("{PUG}compound/cid/{cid}/property/IsomericSMILES/TXT")
}

/// The same SMILES under PubChem's newer property name (2025), asked when [`stereo_url`] gives
/// nothing: `…/compound/cid/<cid>/property/SMILES/TXT`.
pub fn smiles_property_url(cid: u64) -> String {
    format!("{PUG}compound/cid/{cid}/property/SMILES/TXT")
}

/// A name's CIDs: `…/compound/name/<name>/cids/TXT`.
pub fn name_url(name: &str) -> String {
    format!("{PUG}compound/name/{}/cids/TXT", encode(name))
}

// ── the transport ──

/// Why a request got no body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetError {
    /// No network, the network not allowed, DNS, TLS, a connection refused or cut.
    Offline,
    Timeout,
    /// An HTTP answer other than 200 (a 3xx too: redirects are not followed).
    Status(u16),
    /// A body or a URL that is not what was asked for.
    Malformed,
}

/// One HTTPS GET, blocking (the worker thread's).
pub trait Transport: Send + Sync {
    fn get(&self, url: &str, timeout: Duration) -> Result<String, NetError>;
    /// Whether this reaches the network: the real transport does; a test's canned answers do not,
    /// so the no-network rule does not stop them (it stops every transport that says true).
    fn is_network(&self) -> bool {
        true
    }
}

/// The builder's user agent: its name and this build's version, nothing else.
pub fn user_agent() -> String {
    format!("atomic-sandbox-builder/{}", env!("CARGO_PKG_VERSION"))
}

/// `ureq` 2 over rustls (the updater's client, `update::net`): HTTPS only, no redirect followed,
/// the user agent, a timeout per request; only URLs under [`PUG`], and nothing at all while
/// [`network_allowed`] is false.
pub struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    pub fn new() -> UreqTransport {
        UreqTransport { agent: ureq::AgentBuilder::new().user_agent(&user_agent()).https_only(true).redirects(0).timeout_connect(TIMEOUT).build() }
    }
}

impl Transport for UreqTransport {
    fn get(&self, url: &str, timeout: Duration) -> Result<String, NetError> {
        if !network_allowed() {
            return Err(NetError::Offline);
        }
        if !url.starts_with(PUG) || !url.is_ascii() {
            return Err(NetError::Malformed);
        }
        match self.agent.get(url).timeout(timeout).call() {
            Ok(r) if r.status() == 200 => r.into_string().map_err(|e| if timed_out(&e) { NetError::Timeout } else { NetError::Offline }),
            Ok(r) => Err(NetError::Status(r.status())),
            Err(ureq::Error::Status(code, _)) => Err(NetError::Status(code)),
            Err(ureq::Error::Transport(t)) => Err(if timed_out(&t) { NetError::Timeout } else { NetError::Offline }),
        }
    }
}

/// An I/O timeout anywhere in the error's source chain.
fn timed_out(e: &(dyn std::error::Error + 'static)) -> bool {
    let mut cur = Some(e);
    while let Some(e) = cur {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            if matches!(io.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) {
                return true;
            }
        }
        cur = e.source();
    }
    false
}

// ── the clock ──

/// Time for the debounce, the timeout and the rate limit (the tests' is set by hand).
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
    fn sleep(&self, d: Duration);
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d)
    }
}

// ── the cache ──

/// The molecule cache (module doc): records fetched from PubChem, one file per CID. Clones share
/// one memo of what was read, so the worker's [`Cache::store`] is seen by the window's
/// [`Cache::find`].
#[derive(Clone)]
pub struct Cache {
    inner: Arc<CacheInner>,
}

struct CacheInner {
    dir: Option<PathBuf>,
    memo: Mutex<HashMap<(String, i32), Vec<Record>>>,
    /// Every record ([`Cache::all`]), read once until the next store.
    all: Mutex<Option<Vec<Record>>>,
}

impl Cache {
    /// The cache in `dir` (created on the first store; nothing is read until a find).
    pub fn open(dir: PathBuf) -> Cache {
        Cache::with_dir(Some(dir))
    }

    /// No cache (no per-user folder): finds nothing, keeps nothing.
    pub fn none() -> Cache {
        Cache::with_dir(None)
    }

    fn with_dir(dir: Option<PathBuf>) -> Cache {
        Cache { inner: Arc::new(CacheInner { dir, memo: Mutex::new(HashMap::new()), all: Mutex::new(None) }) }
    }

    /// `%LOCALAPPDATA%\atomic-sandbox\molecules`.
    pub fn user_dir() -> Option<PathBuf> {
        crate::paths::user_root().map(|r| r.join("molecules"))
    }

    /// The cached records with this formula and charge (by CID); files that are unreadable, do not
    /// validate or disagree with their names are skipped.
    pub fn find(&self, key: &(String, i32)) -> Vec<Record> {
        let Some(dir) = &self.inner.dir else { return Vec::new() };
        let mut memo = self.inner.memo.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(v) = memo.get(key) {
            return v.clone();
        }
        let prefix = format!("{}_{}_", key.0, key.1);
        let mut found: Vec<(u64, Record)> = Vec::new();
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = e.file_name();
            let Some(cid) = name.to_str().and_then(|n| n.strip_prefix(&prefix)).and_then(|r| r.strip_suffix(".txt")).and_then(|c| c.parse::<u64>().ok()) else { continue };
            if let Some(r) = read_entry(&e.path(), key, cid) {
                found.push((cid, r));
            }
        }
        found.sort_by_key(|f| f.0);
        let v: Vec<Record> = found.into_iter().map(|f| f.1).collect();
        memo.insert(key.clone(), v.clone());
        v
    }

    /// Keeps a PubChem record (it must validate), replacing the file of its CID.
    pub fn store(&self, r: &Record) -> std::io::Result<()> {
        let invalid = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, e);
        let Some(dir) = &self.inner.dir else { return Ok(()) };
        let Source::PubChem { cid } = r.source else { return Err(invalid("only PubChem records are cached".into())) };
        r.validate().map_err(invalid)?;
        let key = r.formula_key();
        std::fs::create_dir_all(dir)?;
        let name = format!("{}_{}_{cid}.txt", key.0, key.1);
        let tmp = dir.join(format!("{name}.tmp"));
        std::fs::write(&tmp, records::write(std::slice::from_ref(r)))?;
        std::fs::rename(&tmp, dir.join(&name))?;
        self.inner.memo.lock().unwrap_or_else(|p| p.into_inner()).remove(&key);
        *self.inner.all.lock().unwrap_or_else(|p| p.into_inner()) = None;
        Ok(())
    }

    /// Every cached record, by CID (the search box's names): each file named
    /// `<formula>_<charge>_<cid>.txt` that holds what its name says, as [`Cache::find`] reads them.
    pub fn all(&self) -> Vec<Record> {
        let Some(dir) = &self.inner.dir else { return Vec::new() };
        let mut all = self.inner.all.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(v) = all.as_ref() {
            return v.clone();
        }
        let mut found: Vec<(u64, Record)> = Vec::new();
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = e.file_name();
            let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".txt")) else { continue };
            let mut parts = stem.rsplitn(3, '_');
            let (Some(cid), Some(q), Some(formula)) = (parts.next(), parts.next(), parts.next()) else { continue };
            let (Ok(cid), Ok(q)) = (cid.parse::<u64>(), q.parse::<i32>()) else { continue };
            if let Some(r) = read_entry(&e.path(), &(formula.to_string(), q), cid) {
                found.push((cid, r));
            }
        }
        found.sort_by_key(|f| f.0);
        let v: Vec<Record> = found.into_iter().map(|f| f.1).collect();
        *all = Some(v.clone());
        v
    }
}

/// One cache file's record, if it is whole and is what its name says.
fn read_entry(path: &std::path::Path, key: &(String, i32), cid: u64) -> Option<Record> {
    let text = String::from_utf8(std::fs::read(path).ok()?).ok()?;
    let mut v = records::parse(&text).ok()?;
    let r = v.pop().filter(|_| v.is_empty())?;
    (r.validate().is_ok() && r.formula_key() == *key && r.source == Source::PubChem { cid }).then_some(r)
}

// ── the worker ──

/// The worker's requests: through the transport, at most one per [`MIN_INTERVAL`], each within its
/// lookup's deadline, and none while the network is not allowed (for a transport that reaches it)
/// or once the job being worked on is stale (review fix 1: a newer drawing generation exists, so
/// the lookup stops between its requests and the queue behind it moves on; Task 5 review fix 1: a
/// search stops the same way once a newer search was asked or the window's online setting went
/// off, [`Lookups::stop_searches`]).
struct Net {
    t: Box<dyn Transport>,
    clock: Arc<dyn Clock>,
    last: Option<Instant>,
    /// The newest drawing generation the window has seen ([`Lookups::supersede`]).
    newest: Arc<AtomicU64>,
    /// The newest search id handed out, or the mark [`Lookups::stop_searches`] set past it.
    newest_search: Arc<AtomicU64>,
    job: Working,
}

/// What the worker is on: a drawn fragment's lookup at a drawing generation, or a search by id.
#[derive(Clone, Copy)]
enum Working {
    Lookup(u64),
    Search(u64),
}

impl Net {
    fn reachable(&self) -> bool {
        !self.t.is_network() || network_allowed()
    }

    /// The job being worked on is superseded: a lookup for an older drawing than the window's, or
    /// a search older than the newest one (or stopped).
    fn stale(&self) -> bool {
        match self.job {
            Working::Lookup(generation) => generation < self.newest.load(Ordering::SeqCst),
            Working::Search(id) => id < self.newest_search.load(Ordering::SeqCst),
        }
    }

    fn get(&mut self, url: &str, deadline: Instant) -> Result<String, NetError> {
        if !self.reachable() || self.stale() {
            return Err(NetError::Offline);
        }
        if let Some(last) = self.last {
            let wait = (last + MIN_INTERVAL).saturating_duration_since(self.clock.now());
            if !wait.is_zero() {
                self.clock.sleep(wait);
            }
        }
        let now = self.clock.now();
        let left = deadline.saturating_duration_since(now);
        if left.is_zero() {
            return Err(NetError::Timeout);
        }
        self.last = Some(now);
        self.t.get(url, left.min(TIMEOUT))
    }
}

/// PubChem has no such thing: a 4xx other than a request timeout or a rate limit. Anything else is
/// the network's or the service's failure.
fn absent(e: NetError) -> bool {
    matches!(e, NetError::Status(s) if (400..500).contains(&s) && s != 408 && s != 429)
}

/// A `…/cids/TXT` body: one CID a line (0, PubChem's "none", and repeats left out).
fn parse_cids(body: &str) -> Result<Vec<u64>, NetError> {
    let mut v = Vec::new();
    for l in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let c: u64 = l.parse().map_err(|_| NetError::Malformed)?;
        if c > 0 && !v.contains(&c) {
            v.push(c);
        }
    }
    Ok(v)
}

/// The CID's 3D conformer as a record (name, layout and stereo still to come, [`finish`]): None when
/// PubChem has no 3D record for it or its SDF is not one we read.
fn fetch_3d(net: &mut Net, cid: u64, deadline: Instant) -> Result<Option<Record>, NetError> {
    let body = match net.get(&sdf_url(cid, "3d"), deadline) {
        Ok(b) => b,
        Err(e) if absent(e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Ok((z, charge, bonds, xyz)) = parse_sdf(&body) else { return Ok(None) };
    let xy = canvas_layout(&xyz, &bonds);
    Ok(Some(Record { name: hill(&z), synonyms: Vec::new(), source: Source::PubChem { cid }, z, charge, bonds, xyz, xy, stereo_defined: true }))
}

/// Completes a fetched record: the canvas layout from the 2D SDF (mapped atom for atom onto the 3D
/// record), the title, and the stereo flag from the isomeric SMILES. Each is best effort: without
/// an answer the layout stays the 3D shape's projection, the name the Hill formula, and the record
/// is taken as stereo-specific (the footer's note then says the stereo is the database's, which is
/// true either way). Whether every answer came (only a whole record is cached; PubChem's 404 counts
/// as an answer).
fn finish(net: &mut Net, mut r: Record, cid: u64, deadline: Instant) -> (Record, bool) {
    let mut complete = true;
    match net.get(&sdf_url(cid, "2d"), deadline) {
        Ok(b) => match parse_sdf(&b).ok().and_then(|two| layout_2d(&r, two)) {
            Some(xy) => r.xy = xy,
            None => complete = false,
        },
        Err(e) if absent(e) => {}
        Err(_) => complete = false,
    }
    match net.get(&title_url(cid), deadline) {
        Ok(b) => {
            let t = b.lines().next().unwrap_or("").trim();
            if !t.is_empty() && t.len() <= 200 && !t.chars().any(char::is_control) {
                r.name = t.to_string();
            } else {
                complete = false;
            }
        }
        Err(e) if absent(e) => {}
        Err(_) => complete = false,
    }
    // the isomeric SMILES, then (minor 1) PubChem's newer name for the same property
    let stereo = [stereo_url(cid), smiles_property_url(cid)].into_iter().find_map(|url| match net.get(&url, deadline) {
        Ok(b) if !b.trim().is_empty() => Some(b.trim().contains(['@', '/', '\\'])),
        _ => None,
    });
    match stereo {
        Some(s) => r.stereo_defined = s,
        None => {
            complete = false;
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::SeqCst) {
                eprintln!("builder: PubChem gave no SMILES for CID {cid} (IsomericSMILES nor SMILES); its shape is taken as stereo-specific");
            }
        }
    }
    (r, complete)
}

/// The 2D SDF's coordinates in `r`'s atom order, through the exact mapping of its graph onto
/// `r`'s, laid out by the canvas rule; None when the graphs differ.
fn layout_2d(r: &Record, (z, charge, bonds, p): (Vec<u8>, Vec<i8>, Vec<(usize, usize, u8)>, Vec<[f64; 3]>)) -> Option<Vec<[f32; 2]>> {
    let d = Drawing { atoms: z.iter().zip(&charge).map(|(&z, &charge)| Atom { z, pos: [0.0; 2], charge }).collect(), bonds: bonds.iter().map(|&(a, b, order)| Bond { a, b, order }).collect() };
    let all: Vec<usize> = (0..d.atoms.len()).collect();
    let MapResult::Mapped(m) = iso::map_fragment(&d, &all, r) else { return None };
    let mut q = vec![[0.0; 3]; r.z.len()];
    for (i, &k) in m.iter().enumerate() {
        q[k] = p[i];
    }
    Some(canvas_layout(&q, &r.bonds))
}

/// One fragment's lookup (module doc): the cache, then (network allowed) PubChem.
fn resolve(net: &mut Net, cache: &Cache, d: &Drawing, frag: &[usize]) -> Lookup {
    let key = iso::drawing_key(d, frag);
    for r in cache.find(&key) {
        if let MapResult::Mapped(m) = iso::map_fragment(d, frag, &r) {
            return Lookup::Found(r, m);
        }
    }
    if !net.reachable() {
        return Lookup::Offline;
    }
    // no record may hold it (spec §5): nothing to ask
    if frag.len() > MAX_ATOMS || frag.iter().any(|&i| !(1..=crate::ui::ptable::Z_ENABLED).contains(&d.atoms[i].z)) {
        return Lookup::NotFound;
    }
    let deadline = net.clock.now() + TIMEOUT;
    let cids = match net.get(&smiles_url(&smiles(d, frag)), deadline).and_then(|b| parse_cids(&b)) {
        Ok(c) => c,
        Err(e) if absent(e) => return Lookup::NotFound,
        Err(_) => return Lookup::Offline,
    };
    for cid in cids.into_iter().take(MAX_CIDS) {
        let r = match fetch_3d(net, cid, deadline) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(_) => return Lookup::Offline,
        };
        if r.formula_key() != key {
            continue;
        }
        if let MapResult::Mapped(m) = iso::map_fragment(d, frag, &r) {
            let (r, complete) = finish(net, r, cid, deadline);
            if r.validate().is_err() {
                continue;
            }
            if complete {
                let _ = cache.store(&r);
            }
            return Lookup::Found(r, m);
        }
    }
    Lookup::NotFound
}

/// An online search (`search.rs`): a name's or a CID's compounds, the first [`MAX_SEARCH`] with a
/// 3D record, each kept in the cache. A formula is asked as a name (PubChem's synonyms hold the
/// common ones: "H2O", "CO2"). A name longer than [`MAX_NAME`] is not sent (no results). A 4xx on
/// the name (PubChem's "no such name", or its refusal of an encoded `/`) is no results, not an
/// error.
fn search(net: &mut Net, cache: &Cache, q: &Query) -> Result<Vec<Hit>, NetError> {
    match q {
        Query::Empty => return Ok(Vec::new()),
        Query::Name(s) | Query::Formula(s) if s.len() > MAX_NAME => return Ok(Vec::new()),
        _ => {}
    }
    if !net.reachable() {
        return Err(NetError::Offline);
    }
    let deadline = net.clock.now() + TIMEOUT;
    let cids = match q {
        Query::Cid(c) => vec![*c],
        Query::Name(s) | Query::Formula(s) => match net.get(&name_url(s), deadline).and_then(|b| parse_cids(&b)) {
            Ok(c) => c,
            Err(e) if absent(e) => return Ok(Vec::new()),
            Err(e) => return Err(e),
        },
        Query::Empty => unreachable!(),
    };
    let mut hits = Vec::new();
    for cid in cids.into_iter().take(MAX_SEARCH) {
        let r = match fetch_3d(net, cid, deadline) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(e) if hits.is_empty() => return Err(e),
            Err(_) => break,
        };
        let (r, complete) = finish(net, r, cid, deadline);
        if r.validate().is_err() {
            continue;
        }
        if complete {
            let _ = cache.store(&r);
        }
        hits.push(Hit { name: r.name.clone(), formula: hill(&r.z), source: "PubChem", record: Some(r) });
    }
    Ok(hits)
}

enum Job {
    Lookup { generation: u64, drawing: Drawing, frag: Vec<usize> },
    Search { id: u64, query: Query },
}

/// A fragment's answer: (drawing generation, fragment, lookup).
type Answer = (u64, Vec<usize>, Lookup);

/// The lookup thread (module doc): requests keyed by (drawing generation, fragment), answered in
/// order. A lookup older than the newest generation (asked for, or announced by
/// [`Lookups::supersede`]) is skipped unanswered, and one running when a newer generation comes
/// stops before its next request, also unanswered.
pub struct Lookups {
    jobs: mpsc::Sender<Job>,
    answers: mpsc::Receiver<Answer>,
    hits: mpsc::Receiver<(u64, Result<Vec<Hit>, NetError>)>,
    latest: Arc<AtomicU64>,
    searches: Arc<AtomicU64>,
    network: bool,
}

impl Lookups {
    pub fn new(t: Box<dyn Transport>, cache: Cache) -> Lookups {
        Lookups::with_clock(t, cache, Arc::new(SystemClock))
    }

    pub fn with_clock(t: Box<dyn Transport>, cache: Cache, clock: Arc<dyn Clock>) -> Lookups {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (answer_tx, answers) = mpsc::channel();
        let (hit_tx, hits) = mpsc::channel();
        let (latest, searches) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let network = t.is_network();
        let mut net = Net { t, clock, last: None, newest: latest.clone(), newest_search: searches.clone(), job: Working::Lookup(0) };
        // without a thread every request goes unanswered, and the session's timeout makes it offline
        let _ = std::thread::Builder::new().name("builder-lookups".into()).spawn(move || {
            for job in job_rx.iter() {
                let sent = match job {
                    Job::Lookup { generation, drawing, frag } => {
                        net.job = Working::Lookup(generation);
                        if net.stale() {
                            continue;
                        }
                        let answer = resolve(&mut net, &cache, &drawing, &frag);
                        // a lookup that went stale while it ran stopped early: its answer is not one
                        if net.stale() {
                            continue;
                        }
                        answer_tx.send((generation, frag, answer)).is_ok()
                    }
                    Job::Search { id, query } => {
                        net.job = Working::Search(id);
                        if net.stale() {
                            continue;
                        }
                        hit_tx.send((id, search(&mut net, &cache, &query))).is_ok()
                    }
                };
                if !sent {
                    break;
                }
            }
        });
        Lookups { jobs, answers, hits, latest, searches, network }
    }

    /// Asks for one fragment of the drawing at `generation`.
    pub fn request(&self, generation: u64, d: &Drawing, frag: &[usize]) {
        self.latest.fetch_max(generation, Ordering::SeqCst);
        let _ = self.jobs.send(Job::Lookup { generation, drawing: d.clone(), frag: frag.to_vec() });
    }

    /// The window's drawing is now at `generation`: lookups for older ones stop (between requests)
    /// or are skipped, so the new drawing's do not wait behind them.
    pub fn supersede(&self, generation: u64) {
        self.latest.fetch_max(generation, Ordering::SeqCst);
    }

    /// The answers that arrived (never blocks).
    pub fn poll(&self) -> Vec<(u64, Vec<usize>, Lookup)> {
        self.answers.try_iter().collect()
    }

    /// Stops every search asked so far: one running makes no further request (its answer, if
    /// any, says offline), and one queued is skipped. The window's online setting going off.
    pub fn stop_searches(&self) {
        self.searches.fetch_add(1, Ordering::SeqCst);
    }

    /// Starts an online search; returns its id (later searches supersede it: one still running
    /// stops before its next request).
    pub fn search(&self, q: &Query) -> u64 {
        let id = self.searches.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.jobs.send(Job::Search { id, query: q.clone() });
        id
    }

    /// The searches answered (never blocks): (id, hits or why none).
    pub fn poll_search(&self) -> Vec<(u64, Result<Vec<Hit>, NetError>)> {
        self.hits.try_iter().collect()
    }

    /// Whether a request can be answered from PubChem: always for a test's canned transport, while
    /// [`network_allowed`] for the real one.
    pub fn reachable(&self) -> bool {
        !self.network || network_allowed()
    }

    /// Blocks up to `t` for answers (tests).
    #[cfg(test)]
    pub(crate) fn wait(&self, t: Duration) -> Vec<Answer> {
        match self.answers.recv_timeout(t) {
            Ok(a) => std::iter::once(a).chain(self.answers.try_iter()).collect(),
            Err(_) => Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn wait_search(&self, t: Duration) -> Option<(u64, Result<Vec<Hit>, NetError>)> {
        self.hits.recv_timeout(t).ok()
    }
}

// ── the record source ──

/// The lookup order (spec §6.3): the shipped list, then the cache, then this generation's online
/// answers (pending, found, not found or offline); a fragment none of them knows is not found.
pub struct Layered<'a> {
    pub lib: &'a Library,
    pub cache: Option<&'a Cache>,
    pub results: &'a HashMap<Vec<usize>, Lookup>,
}

impl RecordSource for Layered<'_> {
    fn find(&self, d: &Drawing, frag: &[usize]) -> Lookup {
        if let found @ Lookup::Found(..) = LocalOnly(self.lib).find(d, frag) {
            return found;
        }
        if let Some(cache) = self.cache {
            for r in cache.find(&iso::drawing_key(d, frag)) {
                if let MapResult::Mapped(m) = iso::map_fragment(d, frag, &r) {
                    return Lookup::Found(r, m);
                }
            }
        }
        self.results.get(frag).cloned().unwrap_or(Lookup::NotFound)
    }
}

// ── one builder's lookups ──

/// The drawing's graph: what decides which records map (an atom moved changes no answer).
type GraphKey = (Vec<(u8, i8)>, Vec<(usize, usize, u8)>);

fn graph_key(d: &Drawing) -> GraphKey {
    let mut bonds: Vec<(usize, usize, u8)> = d.bonds.iter().map(|b| (b.a.min(b.b), b.a.max(b.b), b.order)).collect();
    bonds.sort_unstable();
    (d.atoms.iter().map(|a| (a.z, a.charge)).collect(), bonds)
}

/// One builder's online state (module doc): the drawing generation, each unknown fragment's
/// answer for it, and when each was asked. [`Session::tick`] runs every frame the window shows.
pub struct Session {
    clock: Arc<dyn Clock>,
    lookups: Option<Lookups>,
    cache: Option<Cache>,
    /// The lookups and cache were given (tests): the real ones are never made.
    given: bool,
    graph: Option<GraphKey>,
    enabled: bool,
    generation: u64,
    changed_at: Instant,
    /// Goes up whenever an answer [`Session::layered`] gives changes (the window's shape cache key).
    version: u64,
    /// The fragments of two or more atoms this generation, and whether the shipped list or the
    /// cache knows each.
    local: Vec<(Vec<usize>, bool)>,
    results: HashMap<Vec<usize>, Lookup>,
    requested: HashMap<Vec<usize>, Instant>,
    inbox: Vec<Answer>,
    /// Search answers a test waited for, kept for the next [`Session::poll_search`].
    search_inbox: Vec<(u64, Result<Vec<Hit>, NetError>)>,
}

impl Default for Session {
    fn default() -> Session {
        Session::new(Arc::new(SystemClock), None, None, false)
    }
}

impl Session {
    fn new(clock: Arc<dyn Clock>, lookups: Option<Lookups>, cache: Option<Cache>, given: bool) -> Session {
        let changed_at = clock.now();
        Session { clock, lookups, cache, given, graph: None, enabled: false, generation: 0, changed_at, version: 0, local: Vec::new(), results: HashMap::new(), requested: HashMap::new(), inbox: Vec::new(), search_inbox: Vec::new() }
    }

    /// A session over the given lookups, cache and clock (tests).
    #[cfg(test)]
    pub(crate) fn with(lookups: Lookups, cache: Cache, clock: Arc<dyn Clock>) -> Session {
        Session::new(clock, Some(lookups), Some(cache), true)
    }

    /// The real lookups and cache, made once the process may use the network (never in a test
    /// build, a replay or a screenshot: those see the shipped list alone).
    fn ensure(&mut self) {
        if !self.given && self.lookups.is_none() && network_allowed() {
            let cache = Cache::user_dir().map_or_else(Cache::none, Cache::open);
            self.lookups = Some(Lookups::new(Box::new(UreqTransport::new()), cache.clone()));
            self.cache = Some(cache);
        }
    }

    /// Brings the answers up to date for drawing `d` with the online setting `enabled`: a new
    /// generation when the graph (or the setting) changed (announced to the lookup thread, whose
    /// older lookups then stop), this generation's answers taken in (older ones dropped), each
    /// unknown fragment asked for once the graph has been still for [`DEBOUNCE`], and one
    /// unanswered after [`TIMEOUT`] marked offline — Insert need not wait longer — until its answer
    /// comes after all (it may have queued behind another fragment's lookup), which then relabels
    /// it. Without the setting or the network every unknown fragment is offline at once, and
    /// nothing is asked.
    pub fn tick(&mut self, d: &Drawing, enabled: bool) {
        self.ensure();
        let now = self.clock.now();
        let g = graph_key(d);
        if self.graph.as_ref() != Some(&g) || self.enabled != enabled {
            self.generation += 1;
            self.graph = Some(g);
            self.enabled = enabled;
            self.changed_at = now;
            self.results.clear();
            self.requested.clear();
            self.version += 1;
            let none = HashMap::new();
            let here = Layered { lib: Library::shipped(), cache: self.cache.as_ref(), results: &none };
            self.local = iso::fragments(d).into_iter().filter(|f| f.len() >= 2).map(|f| (here.find(d, &f).is_found(), f)).map(|(k, f)| (f, k)).collect();
            if let Some(l) = &self.lookups {
                l.supersede(self.generation);
                // the setting off: no search may go on either (Task 5 review fix 1)
                if !enabled {
                    l.stop_searches();
                }
            }
        }
        let mut answers = std::mem::take(&mut self.inbox);
        if let Some(l) = &self.lookups {
            answers.extend(l.poll());
        }
        for (generation, frag, answer) in answers {
            // an answer for an older drawing is dropped; one for this drawing is taken even after
            // its timeout marked the fragment offline (review fix 1)
            if generation == self.generation && self.requested.remove(&frag).is_some() {
                self.results.insert(frag, answer);
                self.version += 1;
            }
        }
        let reach = enabled && self.lookups.as_ref().is_some_and(Lookups::reachable);
        let due = now.saturating_duration_since(self.changed_at) >= DEBOUNCE;
        for (frag, known) in &self.local {
            if *known || matches!(self.results.get(frag), Some(Lookup::Found(..) | Lookup::NotFound | Lookup::Offline)) {
                continue;
            }
            let next = if !reach {
                Lookup::Offline
            } else if let Some(&at) = self.requested.get(frag) {
                if now.saturating_duration_since(at) < TIMEOUT {
                    continue;
                }
                // still `requested`: a late answer relabels it
                Lookup::Offline
            } else {
                if due {
                    self.lookups.as_ref().expect("reachable lookups").request(self.generation, d, frag);
                    self.requested.insert(frag.clone(), now);
                }
                Lookup::Pending
            };
            if self.results.get(frag) != Some(&next) {
                self.results.insert(frag.clone(), next);
                self.version += 1;
            }
        }
    }

    /// Changes whenever [`Session::layered`]'s answers do.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// A fragment is waiting for PubChem (or for the debounce): the window keeps repainting.
    pub fn busy(&self) -> bool {
        self.results.values().any(|r| *r == Lookup::Pending)
    }

    /// A timed-out fragment's answer may still come (it was queued behind another lookup): the
    /// window keeps repainting, more slowly, for up to [`LATE_WAIT`] after it was asked.
    pub fn awaiting(&self) -> bool {
        let now = self.clock.now();
        self.requested.values().any(|&at| now.saturating_duration_since(at) < LATE_WAIT)
    }

    /// The record source for the drawing [`Session::tick`] last saw.
    pub fn layered(&self) -> Layered<'_> {
        Layered { lib: Library::shipped(), cache: self.cache.as_ref(), results: &self.results }
    }

    /// The molecule cache the search box reads: the user's, once the process may use the network
    /// (as for the drawn fragments: a test build, a replay or a screenshot sees the shipped list
    /// alone), else none.
    pub fn local_cache(&mut self) -> Cache {
        self.ensure();
        self.cache.clone().unwrap_or_else(Cache::none)
    }

    /// Whether the search box may offer "Search PubChem for '…'": the online setting on and PubChem
    /// reachable (for the app's own transport, [`network_allowed`]; a test's canned one always).
    pub fn can_search(&mut self, enabled: bool) -> bool {
        self.ensure();
        enabled && self.lookups.as_ref().is_some_and(Lookups::reachable)
    }

    /// Starts an online search ([`Lookups::search`]); None when [`Session::can_search`] is false.
    pub fn search(&mut self, q: &Query, enabled: bool) -> Option<u64> {
        if !self.can_search(enabled) {
            return None;
        }
        self.lookups.as_ref().map(|l| l.search(q))
    }

    /// Stops the searches asked so far ([`Lookups::stop_searches`]): the online setting went off.
    pub fn stop_searches(&self) {
        if let Some(l) = &self.lookups {
            l.stop_searches();
        }
    }

    /// The online searches answered since the last call (never blocks).
    pub fn poll_search(&mut self) -> Vec<(u64, Result<Vec<Hit>, NetError>)> {
        let mut v = std::mem::take(&mut self.search_inbox);
        if let Some(l) = &self.lookups {
            v.extend(l.poll_search());
        }
        v
    }

    /// The window's clock (the search's wait).
    pub fn now(&self) -> Instant {
        self.clock.now()
    }

    /// Blocks up to `t` for the worker's next search answer, kept for the next poll (tests).
    #[cfg(test)]
    pub(crate) fn wait_search(&mut self, t: Duration) -> bool {
        match self.lookups.as_ref().and_then(|l| l.wait_search(t)) {
            Some(a) => {
                self.search_inbox.push(a);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Blocks up to `t` for the worker's next answers, kept for the next tick (tests).
    #[cfg(test)]
    pub(crate) fn wait_reply(&mut self, t: Duration) -> bool {
        let got = self.lookups.as_ref().map(|l| l.wait(t)).unwrap_or_default();
        let any = !got.is_empty();
        self.inbox.extend(got);
        any
    }
}

impl Lookup {
    fn is_found(&self) -> bool {
        matches!(self, Lookup::Found(..))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::chem;
    use super::super::shape::{footer_label, shape, tests::draw_record};
    use super::super::Builder;
    use super::*;
    use std::sync::Condvar;
    use units::elements::z_of_symbol;

    /// Plenty for a worker thread to answer canned responses, however loaded the machine (a pass
    /// returns at once).
    const WAIT: Duration = Duration::from_secs(30);

    // ── test transports and clock ──

    /// Answers from a URL → body map (anything else: 404), counting every call. Not the network.
    #[derive(Clone)]
    pub(crate) struct Canned {
        pub map: HashMap<String, Result<String, NetError>>,
        pub otherwise: Result<String, NetError>,
        pub log: Arc<Mutex<Vec<String>>>,
    }

    impl Canned {
        pub(crate) fn new(map: HashMap<String, Result<String, NetError>>) -> Canned {
            Canned { map, otherwise: Err(NetError::Status(404)), log: Arc::default() }
        }
        pub(crate) fn calls(&self, url: &str) -> usize {
            self.log.lock().unwrap().iter().filter(|u| *u == url).count()
        }
        pub(crate) fn total(&self) -> usize {
            self.log.lock().unwrap().len()
        }
    }

    impl Transport for Canned {
        fn get(&self, url: &str, _: Duration) -> Result<String, NetError> {
            self.log.lock().unwrap().push(url.to_string());
            self.map.get(url).cloned().unwrap_or_else(|| self.otherwise.clone())
        }
        fn is_network(&self) -> bool {
            false
        }
    }

    /// Records every URL asked of the transport it wraps.
    pub(crate) struct Recording<T: Transport> {
        pub inner: T,
        pub urls: Arc<Mutex<Vec<String>>>,
    }

    impl<T: Transport> Transport for Recording<T> {
        fn get(&self, url: &str, t: Duration) -> Result<String, NetError> {
            self.urls.lock().unwrap().push(url.to_string());
            self.inner.get(url, t)
        }
        fn is_network(&self) -> bool {
            self.inner.is_network()
        }
    }

    /// Stands for the real network: any call is a failure (it is flagged, then panics).
    pub(crate) struct FailOnCall {
        pub called: Arc<AtomicBool>,
    }

    impl Transport for FailOnCall {
        fn get(&self, url: &str, _: Duration) -> Result<String, NetError> {
            self.called.store(true, Ordering::SeqCst);
            panic!("the network was called: {url}");
        }
    }

    /// [`Canned`] behind a gate: every call waits until the test opens it.
    pub(crate) struct Gate {
        pub inner: Canned,
        pub open: Arc<(Mutex<bool>, Condvar)>,
    }

    pub(crate) fn open_gate(g: &Arc<(Mutex<bool>, Condvar)>) {
        *g.0.lock().unwrap() = true;
        g.1.notify_all();
    }

    impl Transport for Gate {
        fn get(&self, url: &str, t: Duration) -> Result<String, NetError> {
            let mut open = self.open.0.lock().unwrap();
            while !*open {
                open = self.open.1.wait(open).unwrap();
            }
            drop(open);
            self.inner.get(url, t)
        }
        fn is_network(&self) -> bool {
            false
        }
    }

    /// Time set by hand; `sleep` returns at once and is only noted.
    pub(crate) struct ManualClock {
        base: Instant,
        at: Mutex<Duration>,
        pub slept: Mutex<Vec<Duration>>,
    }

    impl ManualClock {
        pub(crate) fn new() -> Arc<ManualClock> {
            Arc::new(ManualClock { base: Instant::now(), at: Mutex::new(Duration::ZERO), slept: Mutex::default() })
        }
        pub(crate) fn advance(&self, d: Duration) {
            *self.at.lock().unwrap() += d;
        }
    }

    impl Clock for ManualClock {
        fn now(&self) -> Instant {
            self.base + *self.at.lock().unwrap()
        }
        fn sleep(&self, d: Duration) {
            self.slept.lock().unwrap().push(d);
        }
    }

    /// A fresh, empty folder for one test's cache.
    pub(crate) fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("atomic-builder-online-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    // ── canned PubChem answers for ethane (CID 6324) ──

    /// A V2000 SDF of these atoms and bonds (atoms from 0), in PubChem's column layout.
    fn sdf(atoms: &[(&str, [f64; 3])], bonds: &[(usize, usize, u8)]) -> String {
        let mut s = format!("6324\n  -OEChem-test\n\n{:3}{:3}  0     0  0  0  0  0  0999 V2000\n", atoms.len(), bonds.len());
        for (sym, p) in atoms {
            s += &format!("{:10.4}{:10.4}{:10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n", p[0], p[1], p[2], sym);
        }
        for &(a, b, o) in bonds {
            s += &format!("{:3}{:3}{:3}  0  0  0  0\n", a + 1, b + 1, o);
        }
        s + "M  END\n$$$$\n"
    }

    /// Ethane's 3D answer, written for the tests in the shape of PubChem's (staggered, C–C 1.512 Å,
    /// C–H 1.094 Å); not fetched.
    fn ethane_3d() -> String {
        let atoms = [
            ("C", [-0.756, 0.0, 0.0]),
            ("C", [0.756, 0.0, 0.0]),
            ("H", [-1.1404, 0.6586, 0.7845]),
            ("H", [-1.1404, 0.3501, -0.9626]),
            ("H", [-1.1405, -1.0087, 0.1781]),
            ("H", [1.1404, -0.3501, 0.9626]),
            ("H", [1.1405, 1.0087, -0.1781]),
            ("H", [1.1404, -0.6586, -0.7845]),
        ];
        sdf(&atoms, &[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (1, 5, 1), (1, 6, 1), (1, 7, 1)])
    }

    /// Ethane's 2D answer, its atoms in another order than the 3D one's (the second carbon first).
    fn ethane_2d() -> String {
        let atoms = [
            ("C", [3.403, 0.5, 0.0]),
            ("C", [2.5369, 0.0, 0.0]),
            ("H", [3.403, 1.12, 0.0]),
            ("H", [3.9399, 0.19, 0.0]),
            ("H", [2.0, 0.31, 0.0]),
            ("H", [2.8469, -0.5369, 0.0]),
            ("H", [2.2269, -0.5369, 0.0]),
            ("H", [3.9, 0.81, 0.0]),
        ];
        sdf(&atoms, &[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 7, 1), (1, 4, 1), (1, 5, 1), (1, 6, 1)])
    }

    pub(crate) fn ethane_answers(stereo: &str) -> HashMap<String, Result<String, NetError>> {
        HashMap::from([
            (smiles_url("[CH3][CH3]"), Ok("6324\n".to_string())),
            (sdf_url(6324, "3d"), Ok(ethane_3d())),
            (sdf_url(6324, "2d"), Ok(ethane_2d())),
            (title_url(6324), Ok("Ethane\n".to_string())),
            (stereo_url(6324), Ok(format!("{stereo}\n"))),
        ])
    }

    /// Ethane drawn: C–C then Fill hydrogens, or (`h_first`) its hydrogens drawn before the carbons.
    pub(crate) fn ethane(h_first: bool) -> Drawing {
        let mut d = Drawing::default();
        if h_first {
            for i in 0..6 {
                d.add_atom(1, [i as f32, 1.0]);
            }
        }
        let c1 = d.add_atom(6, [0.0, 0.0]);
        let c2 = d.add_atom(6, [1.0, 0.0]);
        d.bond(c1, c2);
        if h_first {
            for i in 0..6 {
                d.bond(if i < 3 { c1 } else { c2 }, i);
            }
        } else {
            chem::fill_hydrogens(&mut d);
        }
        d
    }

    fn all(d: &Drawing) -> Vec<usize> {
        (0..d.atoms.len()).collect()
    }

    // ── a minimal SMILES reader, for the round trip only ──

    /// Reads what [`smiles`] writes (bracket atoms with H counts and charges, `-` `=` `#`,
    /// branches, ring digits and `%nn`, `.`): the elements, charges and bonds, each counted
    /// hydrogen an atom of its own bonded to its atom.
    fn read_smiles(s: &str) -> (Vec<u8>, Vec<i8>, Vec<(usize, usize, u8)>) {
        let c: Vec<char> = s.chars().collect();
        let (mut z, mut q, mut bonds, mut hs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut prev, mut stack, mut pending): (Option<usize>, Vec<usize>, u8) = (None, Vec::new(), 1);
        let mut rings: HashMap<u32, (usize, u8)> = HashMap::new();
        let mut ring = |n: u32, prev: Option<usize>, pending: &mut u8, bonds: &mut Vec<(usize, usize, u8)>| {
            let a = prev.expect("a ring digit after an atom");
            match rings.remove(&n) {
                Some((b, o)) => bonds.push((b, a, if *pending != 1 { *pending } else { o })),
                None => {
                    rings.insert(n, (a, *pending));
                }
            }
            *pending = 1;
        };
        let mut i = 0;
        while i < c.len() {
            match c[i] {
                '[' => {
                    let end = i + c[i..].iter().position(|&x| x == ']').expect("a closing bracket");
                    let inner: String = c[i + 1..end].iter().collect();
                    let k = if inner.len() > 1 && inner.as_bytes()[1].is_ascii_lowercase() { 2 } else { 1 };
                    let mut rest = &inner[k..];
                    let mut h = 0;
                    if let Some(r) = rest.strip_prefix('H') {
                        let digits: String = r.chars().take_while(char::is_ascii_digit).collect();
                        h = if digits.is_empty() { 1 } else { digits.parse().unwrap() };
                        rest = &r[digits.len()..];
                    }
                    let charge: i8 = match rest {
                        "" => 0,
                        "+" => 1,
                        "-" => -1,
                        r => r.parse().unwrap_or_else(|_| panic!("charge {r:?} in {s}")),
                    };
                    let a = z.len();
                    z.push(z_of_symbol(&inner[..k]).unwrap_or_else(|| panic!("element {:?} in {s}", &inner[..k])));
                    q.push(charge);
                    hs.push(h);
                    if let Some(p) = prev {
                        bonds.push((p, a, pending));
                    }
                    pending = 1;
                    prev = Some(a);
                    i = end + 1;
                    continue;
                }
                '-' => pending = 1,
                '=' => pending = 2,
                '#' => pending = 3,
                '(' => stack.push(prev.expect("a branch after an atom")),
                ')' => prev = stack.pop(),
                '.' => prev = None,
                '%' => {
                    let n: u32 = c[i + 1..i + 3].iter().collect::<String>().parse().unwrap();
                    ring(n, prev, &mut pending, &mut bonds);
                    i += 3;
                    continue;
                }
                d if d.is_ascii_digit() => ring(d.to_digit(10).unwrap(), prev, &mut pending, &mut bonds),
                other => panic!("unexpected {other:?} in {s}"),
            }
            i += 1;
        }
        assert!(rings.is_empty(), "an unclosed ring in {s}");
        for (a, &h) in hs.iter().enumerate() {
            for _ in 0..h {
                let k = z.len();
                z.push(1);
                q.push(0);
                bonds.push((a, k, 1));
            }
        }
        (z, q, bonds)
    }

    /// `smiles(d, frag)` reads back as the fragment's graph exactly: an isomorphism keeping
    /// elements, charges and every bond's order (not only up to Kekulé forms).
    fn assert_round_trip(d: &Drawing, frag: &[usize]) -> String {
        let s = smiles(d, frag);
        let (z, charge, bonds) = read_smiles(&s);
        let n = z.len();
        let r = Record { name: "read".into(), synonyms: vec![], source: Source::PubChem { cid: 1 }, z, charge, bonds, xyz: vec![[0.0; 3]; n], xy: vec![[0.0; 2]; n], stereo_defined: false };
        let MapResult::Mapped(m) = iso::map_fragment(d, frag, &r) else { panic!("{s} does not read back as the drawing") };
        let pos = |a: usize| frag.iter().position(|&x| x == a).unwrap();
        for b in d.bonds.iter().filter(|b| frag.contains(&b.a) && frag.contains(&b.b)) {
            let (p, q) = (m[pos(b.a)], m[pos(b.b)]);
            let o = r.bonds.iter().find(|&&(x, y, _)| (x, y) == (p, q) || (y, x) == (p, q)).expect("the bond read back").2;
            assert_eq!(o, b.order, "{s}: bond {}–{}", b.a, b.b);
        }
        s
    }

    fn seed(name: &str) -> &'static Record {
        Library::shipped().records().iter().find(|r| r.name == name).unwrap()
    }

    #[test]
    fn smiles_of_the_seed_molecules() {
        // ethanol drawn in a shuffled order, NH₄⁺, benzene (Kekulé), OH⁻
        for (name, perm) in [("ethanol", vec![3, 0, 8, 7, 1, 5, 2, 6, 4]), ("ammonium", vec![0, 1, 2, 3, 4]), ("benzene", (0..12).collect()), ("hydroxide", vec![1, 0])] {
            let r = seed(name);
            let mut d = Drawing::default();
            draw_record(&mut d, r, [0.0, 0.0], &perm);
            let s = assert_round_trip(&d, &all(&d));
            match name {
                "ammonium" => assert_eq!(s, "[NH4+]"),
                "hydroxide" => assert_eq!(s, "[OH-]"),
                "benzene" => assert!(s.matches('=').count() == 3 && s.contains('1'), "Kekulé with a ring closure: {s}"),
                _ => assert!(!s.contains('='), "{s}"),
            }
        }
        // hydrogens that are atoms of their own: H₂, H₃⁺ (a ring with a charged H), a radical
        let mut h2 = Drawing::default();
        h2.add_atom(1, [0.0, 0.0]);
        h2.add_atom(1, [1.0, 0.0]);
        h2.bond(0, 1);
        assert_eq!(assert_round_trip(&h2, &[0, 1]), "[H][H]");
        let mut h3 = h2.clone();
        h3.add_atom(1, [0.5, 0.8]);
        h3.bond(1, 2);
        h3.bond(2, 0);
        h3.atoms[0].charge = 1;
        assert_eq!(assert_round_trip(&h3, &[0, 1, 2]), "[H+]1[H][H]1");
        let mut methyl = Drawing::default();
        methyl.add_atom(6, [0.0, 0.0]);
        for i in 0..3 {
            methyl.add_atom(1, [i as f32, 1.0]);
            methyl.bond(0, i + 1);
        }
        assert_eq!(assert_round_trip(&methyl, &[0, 1, 2, 3]), "[CH3]", "a radical stays one");
        // two rings sharing a bond (naphthalene's skeleton with its hydrogens), and a triple bond
        let mut nap = Drawing::default();
        for i in 0..10 {
            nap.add_atom(6, [i as f32, 0.0]);
        }
        for (a, b, o) in [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1), (4, 6, 1), (6, 7, 2), (7, 8, 1), (8, 9, 2), (9, 3, 1)] {
            let k = nap.bond(a, b).unwrap();
            nap.bonds[k].order = o;
        }
        chem::fill_hydrogens(&mut nap);
        assert_round_trip(&nap, &all(&nap));
        let mut hcn = Drawing::default();
        hcn.add_atom(1, [0.0, 0.0]);
        hcn.add_atom(6, [1.0, 0.0]);
        hcn.add_atom(7, [2.0, 0.0]);
        hcn.bond(0, 1);
        let k = hcn.bond(1, 2).unwrap();
        hcn.bonds[k].order = 3;
        assert_eq!(assert_round_trip(&hcn, &[0, 1, 2]), "[CH]#[N]");
    }

    #[test]
    fn the_pubchem_urls_are_percent_encoded() {
        assert_eq!(smiles_url("[CH3][CH3]"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%5BCH3%5D%5BCH3%5D/cids/TXT");
        assert_eq!(smiles_url("[CH]#[N]"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%5BCH%5D%23%5BN%5D/cids/TXT");
        assert_eq!(smiles_url("[NH4+]"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/%5BNH4%2B%5D/cids/TXT");
        assert_eq!(sdf_url(6324, "3d"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/6324/SDF?record_type=3d");
        assert_eq!(smiles_property_url(6324), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/6324/property/SMILES/TXT");
        assert_eq!(sdf_url(6324, "2d"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/6324/SDF?record_type=2d");
        assert_eq!(title_url(6324), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/6324/property/Title/TXT");
        assert_eq!(stereo_url(6324), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/6324/property/IsomericSMILES/TXT");
        assert_eq!(name_url("ethanol; DROP/..?#"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/ethanol%3B%20DROP%2F..%3F%23/cids/TXT");
        assert_eq!(name_url("H₂O"), "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/H%E2%82%82O/cids/TXT");
        assert_eq!(user_agent(), format!("atomic-sandbox-builder/{}", env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn a_drawing_found_online_maps_and_caches() {
        let dir = temp_dir("found");
        let canned = Canned::new(ethane_answers("CC"));
        let clock = ManualClock::new();
        let l = Lookups::with_clock(Box::new(canned.clone()), Cache::open(dir.clone()), clock.clone());
        let d = ethane(false);
        assert_eq!(smiles(&d, &all(&d)), "[CH3][CH3]");
        l.request(1, &d, &all(&d));
        let got = l.wait(WAIT);
        assert_eq!(got.len(), 1);
        let (generation, frag, Lookup::Found(r, m)) = &got[0] else { panic!("{:?}", got[0].2) };
        assert_eq!((*generation, frag.clone()), (1, all(&d)));
        assert_eq!(iso::map_fragment(&d, frag, r), MapResult::Mapped(m.clone()), "the exact mapping");
        assert_eq!((r.name.as_str(), &r.source, r.stereo_defined), ("Ethane", &Source::PubChem { cid: 6324 }, false));
        assert_eq!(r.xyz[0], [-0.756, 0.0, 0.0], "the 3D SDF's shape");
        r.validate().unwrap();
        // the 2D layout, mapped onto the 3D atoms: canvas units, mean bond 1
        let mean = r.bonds.iter().map(|&(a, b, _)| ((r.xy[a][0] - r.xy[b][0]) as f64).hypot((r.xy[a][1] - r.xy[b][1]) as f64)).sum::<f64>() / r.bonds.len() as f64;
        assert!((mean - 1.0).abs() < 1e-6, "{mean}");
        // every call once, a rate-limit wait before each but the first
        for url in ethane_answers("CC").keys() {
            assert_eq!(canned.calls(url), 1, "{url}");
        }
        assert_eq!(*clock.slept.lock().unwrap(), vec![MIN_INTERVAL; 4]);
        // kept: a new lookup thread whose network must not be called answers from the cache, for
        // ethane drawn in another atom order
        let key = ("C2H6".to_string(), 0);
        assert_eq!(Cache::open(dir.clone()).find(&key).len(), 1);
        let called = Arc::new(AtomicBool::new(false));
        let again = Lookups::new(Box::new(FailOnCall { called: called.clone() }), Cache::open(dir.clone()));
        let other = ethane(true);
        again.request(1, &other, &all(&other));
        let got = again.wait(WAIT);
        let [(_, _, Lookup::Found(r2, m2))] = &got[..] else { panic!("{got:?}") };
        assert_eq!(r2, r);
        assert_eq!(iso::map_fragment(&other, &all(&other), r2), MapResult::Mapped(m2.clone()));
        assert!(!called.load(Ordering::SeqCst));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stereo_specific_record_carries_the_note() {
        let dir = temp_dir("stereo");
        let mut b = Builder::default();
        let clock = ManualClock::new();
        let cache = Cache::open(dir.clone());
        b.lookup = Session::with(Lookups::with_clock(Box::new(Canned::new(ethane_answers("[C@H]stand-in"))), cache.clone(), clock.clone()), cache, clock.clone());
        b.drawing = ethane(false);
        b.shaped();
        clock.advance(DEBOUNCE);
        b.shaped();
        assert!(b.lookup.wait_reply(WAIT));
        assert_eq!(footer_label(b.shaped()), "shape: PubChem \"Ethane\" (CID 6324) · stereo as in the database (CID 6324)");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Task 3's concern 5: the window's shape cache follows the answers, not the drawing alone.
    #[test]
    fn an_answer_relabels_the_unchanged_drawing() {
        let dir = temp_dir("relabel");
        let mut b = Builder::default();
        let clock = ManualClock::new();
        let cache = Cache::open(dir.clone());
        let canned = Canned::new(ethane_answers("CC"));
        b.lookup = Session::with(Lookups::with_clock(Box::new(canned.clone()), cache.clone(), clock.clone()), cache, clock.clone());
        b.drawing = ethane(false);
        let drawn = b.drawing.clone();
        assert_eq!(footer_label(b.shaped()), "looking up online…");
        assert_eq!(canned.total(), 0, "not before the debounce");
        clock.advance(DEBOUNCE);
        assert_eq!(footer_label(b.shaped()), "looking up online…");
        assert!(b.lookup.wait_reply(WAIT));
        let s = b.shaped().clone();
        assert_eq!(footer_label(&s), "shape: PubChem \"Ethane\" (CID 6324)");
        assert_eq!(b.drawing, drawn, "no edit");
        assert!(!b.lookup.busy());
        // and the shape is the record's
        let iso::MapResult::Mapped(m) = iso::map_fragment(&drawn, &all(&drawn), &b.lookup.layered().cache.unwrap().find(&("C2H6".into(), 0))[0]) else { panic!() };
        let r = &b.lookup.layered().cache.unwrap().find(&("C2H6".into(), 0))[0];
        let d01 = |x: &[[f64; 3]], i: usize, j: usize| ((x[i][0] - x[j][0]).powi(2) + (x[i][1] - x[j][1]).powi(2) + (x[i][2] - x[j][2]).powi(2)).sqrt();
        let want = units::angstrom_to_bohr(d01(&r.xyz, m[0], m[2]));
        assert!((d01(&s.positions, 0, 2) - want).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_timeout_falls_back_offline() {
        let d = ethane(false);
        let mut timing_out = Canned::new(HashMap::new());
        timing_out.otherwise = Err(NetError::Timeout);
        let l = Lookups::with_clock(Box::new(timing_out), Cache::none(), ManualClock::new());
        l.request(1, &d, &all(&d));
        assert_eq!(l.wait(WAIT)[0].2, Lookup::Offline);
        // the service failing is offline too; PubChem's 404 (no such structure) is not found
        let mut busy = Canned::new(HashMap::new());
        busy.otherwise = Err(NetError::Status(503));
        let l = Lookups::with_clock(Box::new(busy), Cache::none(), ManualClock::new());
        l.request(1, &d, &all(&d));
        assert_eq!(l.wait(WAIT)[0].2, Lookup::Offline);
        let l = Lookups::with_clock(Box::new(Canned::new(HashMap::new())), Cache::none(), ManualClock::new());
        l.request(1, &d, &all(&d));
        assert_eq!(l.wait(WAIT)[0].2, Lookup::NotFound);
        // a CID without a 3D record (404) is not in the database
        let mut no3d = ethane_answers("CC");
        no3d.remove(&sdf_url(6324, "3d"));
        let l = Lookups::with_clock(Box::new(Canned::new(no3d)), Cache::none(), ManualClock::new());
        l.request(1, &d, &all(&d));
        assert_eq!(l.wait(WAIT)[0].2, Lookup::NotFound);
        // through the window's session: the footer reads offline
        let mut timing_out = Canned::new(HashMap::new());
        timing_out.otherwise = Err(NetError::Timeout);
        let clock = ManualClock::new();
        let mut s = Session::with(Lookups::with_clock(Box::new(timing_out), Cache::none(), clock.clone()), Cache::none(), clock.clone());
        s.tick(&d, true);
        clock.advance(DEBOUNCE);
        s.tick(&d, true);
        assert!(s.wait_reply(WAIT));
        s.tick(&d, true);
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: approximate (offline)");
    }

    /// Review focus 1: an answer asked for at generation 1 arrives after the drawing changed. The
    /// changed drawing (the C–C made double: the same atoms, so the stale mapping would even pass
    /// the shape's checks) is neither labelled nor placed by it.
    #[test]
    fn a_late_answer_for_an_old_drawing_is_dropped() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let clock = ManualClock::new();
        let canned = Canned::new(ethane_answers("CC"));
        let l = Lookups::with_clock(Box::new(Gate { inner: canned.clone(), open: gate.clone() }), Cache::none(), clock.clone());
        let mut s = Session::with(l, Cache::none(), clock.clone());
        let d1 = ethane(false);
        s.tick(&d1, true);
        assert_eq!(s.generation(), 1);
        clock.advance(DEBOUNCE);
        s.tick(&d1, true); // asked, held at the gate
        let mut d2 = d1.clone();
        d2.cycle_order(0);
        s.tick(&d2, true);
        assert_eq!(s.generation(), 2);
        // a generation-1 answer the worker sent just before the change: dropped on arrival
        let plain = Lookups::with_clock(Box::new(Canned::new(ethane_answers("CC"))), Cache::none(), ManualClock::new());
        plain.request(1, &d1, &all(&d1));
        let early = plain.wait(WAIT);
        assert!(matches!(early[0].2, Lookup::Found(..)));
        s.inbox.extend(early);
        s.tick(&d2, true);
        let shaped = shape(&d2, &s.layered());
        assert_eq!(footer_label(&shaped), "looking up online…", "generation 2 is still waiting for its own");
        assert_eq!(shaped.positions, super::super::embed::embed(&d2), "placed as the embedding, not as ethane");
        assert!(s.layered().results.values().all(|r| *r == Lookup::Pending));
        // the generation-1 lookup still running stops before its next request (review fix 1): it
        // sends no answer, and generation 2's lookup, queued behind it, is answered next
        open_gate(&gate);
        clock.advance(DEBOUNCE);
        s.tick(&d2, true);
        assert!(s.wait_reply(WAIT));
        assert!(s.inbox.iter().all(|a| a.0 == 2), "{:?}", s.inbox.iter().map(|a| a.0).collect::<Vec<_>>());
        s.tick(&d2, true);
        assert_eq!(footer_label(&shape(&d2, &s.layered())), "shape: approximate (not in the database) — Relax after recommended");
        // (generation 1's lookup was at the gate with its first request, or skipped before it if the
        // thread had not reached it yet: either way it went no further)
        let urls = canned.log.lock().unwrap().clone();
        assert!(!urls.contains(&sdf_url(6324, "3d")), "{urls:?}");
        assert!(urls.iter().filter(|u| **u == smiles_url("[CH3][CH3]")).count() <= 1, "{urls:?}");
        assert_eq!(urls.last(), Some(&smiles_url(&smiles(&d2, &all(&d2)))));
        // a moved atom is the same graph: no new generation
        let mut moved = d2.clone();
        moved.atoms[0].pos = [5.0, 5.0];
        s.tick(&moved, true);
        assert_eq!(s.generation(), 2);
    }

    /// Review fix 1: lookups queue one behind another, so a fragment can pass its 5 s while still
    /// waiting its turn. It reads offline then (Insert need not wait longer), and its answer, when
    /// it comes for the same drawing, relabels it.
    #[test]
    fn a_queued_fragment_past_its_timeout_is_relabelled() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        // the window's clock runs on; the worker's (its lookups' own deadlines) does not, as when
        // a lookup sits queued while the one before it takes its time
        let (clock, worker_clock) = (ManualClock::new(), ManualClock::new());
        let l = Lookups::with_clock(Box::new(Gate { inner: Canned::new(ethane_answers("CC")), open: gate.clone() }), Cache::none(), worker_clock);
        let mut s = Session::with(l, Cache::none(), clock.clone());
        // two ethanes: two unknown fragments, two lookups
        let mut d = ethane(false);
        let second = ethane(true);
        let n = d.atoms.len();
        for a in &second.atoms {
            let i = d.add_atom(a.z, [a.pos[0] + 6.0, a.pos[1]]);
            d.atoms[i].charge = a.charge;
        }
        for b in &second.bonds {
            d.bond(n + b.a, n + b.b);
        }
        assert_eq!(iso::fragments(&d).len(), 2);
        s.tick(&d, true);
        clock.advance(DEBOUNCE);
        s.tick(&d, true);
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: 2 approximate · looking up online…");
        clock.advance(TIMEOUT);
        s.tick(&d, true);
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: 2 approximate");
        assert!(!s.busy() && s.awaiting(), "Insert is free; the window still repaints for the answers");
        open_gate(&gate);
        while s.inbox.len() < 2 {
            assert!(s.wait_reply(WAIT), "both answers come");
        }
        s.tick(&d, true);
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: 2 from PubChem");
        assert!(!s.awaiting());
        // an hour on, nothing asked is left to wait for
        clock.advance(LATE_WAIT);
        assert!(!s.awaiting());
    }

    /// Minor 1: without an answer for `IsomericSMILES`, PubChem's newer `SMILES` property gives the
    /// stereo flag; with neither, the record is taken as stereo-specific and not cached.
    #[test]
    fn the_stereo_flag_falls_back_to_the_smiles_property() {
        let d = ethane(false);
        for (newer, stereo, kept) in [(Some("C[C@H]stand-in"), true, 1), (Some("CC"), false, 1), (None, true, 0)] {
            let dir = temp_dir(&format!("stereo-fallback-{kept}-{stereo}"));
            let mut answers = ethane_answers("unused");
            answers.remove(&stereo_url(6324));
            if let Some(v) = newer {
                answers.insert(smiles_property_url(6324), Ok(format!("{v}\n")));
            }
            let l = Lookups::with_clock(Box::new(Canned::new(answers)), Cache::open(dir.clone()), ManualClock::new());
            l.request(1, &d, &all(&d));
            let got = l.wait(WAIT);
            let Lookup::Found(r, _) = &got[0].2 else { panic!("{:?}", got[0].2) };
            assert_eq!(r.stereo_defined, stereo, "{newer:?}");
            assert_eq!(Cache::open(dir.clone()).find(&("C2H6".into(), 0)).len(), kept, "only a whole record is kept");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn a_corrupt_cache_file_is_refetched() {
        let dir = temp_dir("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let key = ("C2H6".to_string(), 0);
        let file = dir.join("C2H6_0_6324.txt");
        // garbage, bytes that are not text, and a whole record that is not what its name says
        let water = records::write(std::slice::from_ref(seed("water")));
        for bad in [b"ASMOLECULES 1\nmol ethane\natom C 0 0 0".to_vec(), vec![0xff, 0xfe, 0x00, 0x13], water.into_bytes()] {
            std::fs::write(&file, &bad).unwrap();
            assert!(Cache::open(dir.clone()).find(&key).is_empty());
        }
        std::fs::write(&file, b"not a record at all").unwrap();
        let canned = Canned::new(ethane_answers("CC"));
        let cache = Cache::open(dir.clone());
        let l = Lookups::with_clock(Box::new(canned.clone()), cache.clone(), ManualClock::new());
        let d = ethane(false);
        l.request(1, &d, &all(&d));
        assert!(matches!(l.wait(WAIT)[0].2, Lookup::Found(..)));
        assert_eq!(canned.calls(&sdf_url(6324, "3d")), 1, "refetched");
        assert_eq!(cache.find(&key).len(), 1, "the shared memo sees the store");
        let fresh = Cache::open(dir.clone()).find(&key);
        assert_eq!((fresh.len(), fresh[0].name.as_str()), (1, "Ethane"), "the file was replaced");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_network_when_off_replay_or_test() {
        assert!(!network_allowed(), "a test build has no network by default");
        set_network_allowed(false); // what `main` does under --replay and --screenshot
        let d = ethane(false);
        // the setting off: nothing asked, even of a transport that could answer
        let canned = Canned::new(ethane_answers("CC"));
        let clock = ManualClock::new();
        let mut s = Session::with(Lookups::with_clock(Box::new(canned.clone()), Cache::none(), clock.clone()), Cache::none(), clock.clone());
        s.tick(&d, false);
        clock.advance(DEBOUNCE + TIMEOUT);
        s.tick(&d, false);
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: approximate (offline)");
        assert!(!s.busy());
        // the network not allowed: offline at once, and the network is never called
        let called = Arc::new(AtomicBool::new(false));
        let mut s = Session::with(Lookups::with_clock(Box::new(FailOnCall { called: called.clone() }), Cache::none(), clock.clone()), Cache::none(), clock.clone());
        s.tick(&d, true);
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: approximate (offline)");
        clock.advance(DEBOUNCE);
        s.tick(&d, true);
        assert!(!s.busy());
        // the lookup thread refuses as well, for lookups and searches
        let l = Lookups::new(Box::new(FailOnCall { called: called.clone() }), Cache::none());
        assert!(!l.reachable());
        l.request(1, &d, &all(&d));
        assert_eq!(l.wait(WAIT)[0].2, Lookup::Offline);
        l.search(&Query::Name("ethane".into()));
        assert_eq!(l.wait_search(WAIT).unwrap().1, Err(NetError::Offline));
        assert!(!called.load(Ordering::SeqCst));
        assert_eq!(canned.total(), 0);
        // and the real transport, which checks the switch before anything else
        assert_eq!(UreqTransport::new().get(&smiles_url("[CH3][CH3]"), TIMEOUT), Err(NetError::Offline));
        // a builder as the app makes it, in a test build: no lookups made at all
        let mut b = Builder::default();
        b.drawing = d.clone();
        assert_eq!(footer_label(b.shaped()), "shape: approximate (offline)");
        assert!(b.lookup.lookups.is_none() && b.lookup.cache.is_none());
    }

    /// Task 5 review fix 1: the online setting going off supersedes a drawn fragment's lookup in
    /// flight — it makes no further request — and stops the searches too.
    #[test]
    fn the_setting_off_stops_a_drawn_lookup_in_flight() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let urls = Arc::new(Mutex::new(Vec::new()));
        let t = Recording { inner: Gate { inner: Canned::new(ethane_answers("CC")), open: gate.clone() }, urls: urls.clone() };
        let clock = ManualClock::new();
        let mut s = Session::with(Lookups::with_clock(Box::new(t), Cache::none(), ManualClock::new()), Cache::none(), clock.clone());
        let d = ethane(false);
        s.tick(&d, true);
        clock.advance(DEBOUNCE);
        s.tick(&d, true); // asked: the SMILES request is out, held at the gate
        let t0 = Instant::now();
        while urls.lock().unwrap().is_empty() {
            assert!(t0.elapsed() < WAIT, "the worker never asked");
            std::thread::sleep(Duration::from_millis(2));
        }
        let l = s.lookups.as_ref().unwrap();
        let before = l.searches.load(Ordering::SeqCst);
        s.tick(&d, false);
        assert!(s.lookups.as_ref().unwrap().searches.load(Ordering::SeqCst) > before, "the searches are stopped too");
        assert_eq!(footer_label(&shape(&d, &s.layered())), "shape: approximate (offline)");
        open_gate(&gate);
        // a point behind it on the same worker: an empty search answers at once
        let l = s.lookups.as_ref().unwrap();
        l.search(&Query::Empty);
        assert!(l.wait_search(WAIT).is_some());
        assert_eq!(*urls.lock().unwrap(), [smiles_url("[CH3][CH3]")], "nothing after the setting went off");
        assert!(l.wait(Duration::from_millis(50)).is_empty(), "and no answer for the superseded drawing");
    }

    /// R1: the online search, by name or CID, through canned answers; every URL it builds is ASCII
    /// and under PubChem's root.
    #[test]
    fn an_online_search_gives_hits() {
        let dir = temp_dir("search");
        let mut answers = ethane_answers("CC");
        answers.insert(name_url("ethane"), Ok("6324\n".into()));
        let urls = Arc::new(Mutex::new(Vec::new()));
        let l = Lookups::with_clock(Box::new(Recording { inner: Canned::new(answers), urls: urls.clone() }), Cache::open(dir.clone()), ManualClock::new());
        for q in [Query::Name("ethane".into()), Query::Cid(6324)] {
            let id = l.search(&q);
            let (got, hits) = l.wait_search(WAIT).unwrap();
            assert_eq!(got, id);
            let hits = hits.unwrap();
            assert_eq!(hits.len(), 1, "{q:?}");
            assert_eq!((hits[0].name.as_str(), hits[0].formula.as_str(), hits[0].source), ("Ethane", "C2H6", "PubChem"));
            assert_eq!(hits[0].record.as_ref().unwrap().source, Source::PubChem { cid: 6324 });
        }
        let n = urls.lock().unwrap().len();
        assert_eq!(l.search(&Query::Empty), 3);
        assert_eq!(l.wait_search(WAIT).unwrap().1, Ok(vec![]));
        assert_eq!(urls.lock().unwrap().len(), n, "an empty query asks nothing");
        assert!(urls.lock().unwrap().iter().all(|u| u.is_ascii() && u.starts_with(PUG) && !u.contains(' ')));
        assert_eq!(Cache::open(dir.clone()).find(&("C2H6".into(), 0)).len(), 1, "kept");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
