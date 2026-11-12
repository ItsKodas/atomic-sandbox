//! Bonds guessed from experience (user request 2026-11-13). Under the estimated solver the model
//! has no electrons, so between two DFT densities the bonds on screen are the last DFT graph's,
//! drawn stale (dashed and faded) between the current positions — an H₃ that pulls apart for a
//! moment still shows connected. This module remembers what this app's own DFT said about pairs
//! it has seen, and while the graph on screen is stale it guesses each current pair from those
//! verdicts.
//!
//! **Rule 1.** Nothing here is typed in per element: no radii, no valences, no bond lengths. Every
//! example is a verdict of the app's own density detector (`sim::bonds`) on the app's own DFT
//! density, recorded as the app runs; the element enters only as the atomic number Z of the pair,
//! the same way it enters the solver. With no experience every pair is unknown and the display is
//! exactly the stale graph it was before.
//!
//! # Recording ([`examples`], [`Experience::on_snapshot`])
//! Whenever a snapshot carries a **fresh** graph — `bonds_stale` false, the forces fresh (a drag
//! before its re-evaluation shows the old graph with the new positions: not fresh), the graph's
//! nucleus count the scene's — every pair i < j closer than `sim::bonds::R_MAX` becomes one
//! [`Example`]: its (Z, neighbour count) ends sorted together, R (Bohr), and whether the graph
//! holds the pair. An atom's neighbour count is the number of other nuclei within `R_MAX` (the
//! detector's own reach), capped at [`COUNT_CAP`]. The same graph (the same `Arc`, or a graph at
//! bit-for-bit the same nuclei and positions as the last one recorded — a second "Solve Bonds"
//! while paused) is recorded once. Recording does not depend on the display setting. That rule
//! is per session: a launch that solves bonds at a preset's start geometry again appends the same
//! examples again. Harmless for the all-agree rule (the same verdicts), it only adds weight to
//! geometries the app starts from and grows the file a little.
//!
//! # Storage ([`append`], [`open`])
//! With the estimated solver learning into a shared collection (`ModelSource::Collection`), the
//! examples go to `bond_experience.txt` in that collection's folder: the collection's key is the
//! solver surface (grid, σ, functional, ionic potential, tolerance), and a bond verdict depends on
//! that surface. Otherwise they live in memory for the session, keyed by the scene's h, σ, ionic
//! policy, charge, multiplicity and engine ([`Place::Memory`]); a change of any of them starts an
//! empty store. A charged scene never learns into a collection (collections are neutral), so its
//! experience is in memory only.
//!
//! The file is text: a header line `bond-experience 1` (the format version), then one line per
//! example, `za zb ca cb R bonded check` (`check`: FNV-1a of the text before it, 8 hex digits).
//! Lines are appended only, each batch in one write in append mode, so two running copies may
//! append at once and their batches interleave whole; a line that does not parse or whose check
//! fails (a torn last line after a crash, a torn write) is skipped, and a writer that finds the
//! file not ending in a newline starts its batch on a new line. A header naming another version
//! (a newer build) leaves the file alone: nothing is read from it or written to it, and the session
//! keeps its examples in memory. At most [`CAP`] examples are kept: on load the newest. The file is
//! never rewritten except to compact it when it holds more than 2 × [`CAP`] examples: the newest
//! [`CAP`] are written to a temporary file that is renamed over it. Examples appended between the
//! compaction's read and its rename are lost from the file — another copy's, and this session's
//! own, since a compaction this session starts runs on a detached thread while it keeps appending
//! (they stay in this session's memory). The load (and a compaction) runs on a thread; examples
//! recorded meanwhile wait for it, and are lost if the app exits before it finishes. Display data,
//! not physics: a lost example only means a pair is asked about again.
//!
//! # Prediction ([`Store::predict`], [`compose`])
//! For a pair with the same (Z, count) ends, its examples with |R − R_ex| ≤ [`MATCH_R`] are counted
//! (the store is indexed by that key with R sorted and a running count of the bonded ones, so a
//! lookup is two binary searches). At least [`MIN_AGREE`] of them, all agreeing: that verdict; else
//! unknown. Measured (the ignored `three_hundred_atoms_over_a_full_store_measured`, 2026-11-13): 300
//! atoms at liquid-water density, 2 907 pairs within `R_MAX`, over a full store of [`CAP`]
//! examples — the drawing ([`compose`]) 0.55 ms, its lookups 0.29 ms against 2.6 ms for the first
//! version's walk over each window. `MATCH_R` = 0.1 Bohr and `MIN_AGREE` = 3 are **display constants,
//! not physics**: how close and how many before the view trusts what it saw.
//!
//! # Display ([`compose`], `overlay::bond_lines_drawn`)
//! Only under the estimated solver, while the graph on screen is stale, bonds are shown and View ▸
//! Bonds & labels ▸ "guess bonds from experience" is on (otherwise nothing is computed): a pair
//! experience calls bonded is drawn solid in a "from experience" tint — with the last DFT graph's
//! kind and strength when that graph had it, else as a plain shared bond at the middle of the
//! strength scale ([`plain_bond`]); a pair experience calls not bonded is not drawn, even if the
//! last graph had it; an unknown pair keeps the last graph's verdict, drawn stale as before. An
//! angle arc between two guessed bonds is drawn in the same tint (one with a stale bond stays
//! stale), and its label and a guessed bond's length label in the same tint at an alpha between the
//! stale and the fresh ones: nothing guessed looks fresh. The drawing is made again only when the
//! positions, the graph or the store change. Molecule names and labels stay the DFT graph's (they
//! are read on the worker from `DynSnapshot::bonds`), and so do the Selection panel's bond list
//! and numbers: experience only changes the lines, the angle arcs and the length labels drawn.
use sim::BondGraph;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The file in a collection's folder.
pub const FILE_NAME: &str = "bond_experience.txt";
/// The header's word and this build's format version.
pub const HEADER_WORD: &str = "bond-experience";
pub const FORMAT_VERSION: u32 = 1;
/// The most examples kept (newest first).
pub const CAP: usize = 200_000;
/// Display constants, not physics (module doc): the R window (Bohr) and the agreeing examples
/// needed for a verdict.
pub const MATCH_R: f64 = 0.1;
pub const MIN_AGREE: usize = 3;
/// A neighbour count is capped here: past it, more neighbours read as "crowded".
pub const COUNT_CAP: u8 = 6;

/// (Z, neighbour count) of the pair's two ends, the smaller tuple first.
pub type Key = (u8, u8, u8, u8);

/// One recorded verdict of the detector (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Example {
    pub za: u8,
    pub zb: u8,
    pub ca: u8,
    pub cb: u8,
    /// Bohr.
    pub r: f64,
    pub bonded: bool,
}

/// The pair's key from its two ends (Z, count) in either order: the smaller tuple first, so a
/// pair reads the same from both ends and the counts stay with their Z.
pub fn pair_key(zi: u8, ci: u8, zj: u8, cj: u8) -> Key {
    if (zi, ci) <= (zj, cj) {
        (zi, zj, ci, cj)
    } else {
        (zj, zi, cj, ci)
    }
}

fn fnv1a(s: &str) -> u32 {
    s.bytes().fold(0x811c_9dc5u32, |h, b| (h ^ b as u32).wrapping_mul(0x0100_0193))
}

impl Example {
    pub fn key(&self) -> Key {
        (self.za, self.zb, self.ca, self.cb)
    }

    /// The file line, without its newline.
    pub fn line(&self) -> String {
        let body = format!("{} {} {} {} {:.4} {}", self.za, self.zb, self.ca, self.cb, self.r, self.bonded as u8);
        format!("{body} {:08x}", fnv1a(&body))
    }

    /// A line of the file, or `None` when it is not a whole, checked, canonical example.
    pub fn parse(line: &str) -> Option<Example> {
        let line = line.trim_end_matches(['\r', '\n']);
        let (body, check) = line.rsplit_once(' ')?;
        if check.len() != 8 || u32::from_str_radix(check, 16).ok()? != fnv1a(body) {
            return None;
        }
        let f: Vec<&str> = body.split(' ').collect();
        if f.len() != 6 {
            return None;
        }
        let e = Example {
            za: f[0].parse().ok()?,
            zb: f[1].parse().ok()?,
            ca: f[2].parse().ok()?,
            cb: f[3].parse().ok()?,
            r: f[4].parse().ok()?,
            bonded: match f[5] {
                "1" => true,
                "0" => false,
                _ => return None,
            },
        };
        let canonical = pair_key(e.za, e.ca, e.zb, e.cb) == e.key();
        (canonical && e.za > 0 && e.ca <= COUNT_CAP && e.cb <= COUNT_CAP && e.r.is_finite() && e.r > 0.0 && e.r < sim::bonds::R_MAX).then_some(e)
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Each nucleus's count of other nuclei closer than `sim::bonds::R_MAX`, capped at [`COUNT_CAP`].
pub fn neighbour_counts(pos: &[[f64; 3]]) -> Vec<u8> {
    (0..pos.len()).map(|i| (0..pos.len()).filter(|&j| j != i && dist(pos[i], pos[j]) < sim::bonds::R_MAX).count().min(COUNT_CAP as usize) as u8).collect()
}

/// The examples of a fresh `graph` of the nuclei `z` at `pos` (module doc): every pair closer
/// than `R_MAX`, bonded when the graph holds it. Nothing when the counts disagree.
pub fn examples(z: &[u8], pos: &[[f64; 3]], graph: &BondGraph) -> Vec<Example> {
    let n = pos.len();
    if z.len() != n || graph.n_nuclei != n {
        return Vec::new();
    }
    let counts = neighbour_counts(pos);
    let held: HashSet<(usize, usize)> = graph.bonds.iter().map(|b| (b.i.min(b.j), b.i.max(b.j))).collect();
    let bonded = |i: usize, j: usize| held.contains(&(i, j));
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let r = dist(pos[i], pos[j]);
            if !(r > 0.0 && r < sim::bonds::R_MAX) {
                continue;
            }
            let (za, zb, ca, cb) = pair_key(z[i], counts[i], z[j], counts[j]);
            out.push(Example { za, zb, ca, cb, r, bonded: bonded(i, j) });
        }
    }
    out
}

/// What experience says about a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Bonded,
    NotBonded,
    Unknown,
}

/// One key's examples, R sorted after [`Store::prepare`], with the running count of the bonded
/// ones: `bonded_before[k]` is how many of `v[..k]` are bonded (`v.len() + 1` entries).
#[derive(Debug, Default)]
struct Slot {
    v: Vec<(f64, bool, u64)>,
    bonded_before: Vec<u32>,
}

/// The examples, indexed by [`Key`] with R sorted (module doc). Each entry carries its insertion
/// number; the newest `cap` are live. [`Store::prepare`] sweeps the dead ones out of the keys they
/// left (`order` names each entry's key by insertion number), sorts the keys that changed and
/// rebuilds their running counts, so [`Store::predict`] is two binary searches. Without a
/// `prepare` (no stale graph to guess over) [`Store::add`] settles the store itself once an eighth
/// of the cap is dead, so memory stays bounded.
#[derive(Debug)]
pub struct Store {
    index: HashMap<Key, Slot>,
    /// The key of every entry not yet swept, oldest first: `order[k]` is insertion number `swept + k`.
    order: VecDeque<Key>,
    /// Keys added to or left by a dead entry since the last [`Store::prepare`].
    touched: HashSet<Key>,
    next: u64,
    cap: usize,
    swept: u64,
}

impl Default for Store {
    fn default() -> Store {
        Store::with_cap(CAP)
    }
}

impl Store {
    pub fn with_cap(cap: usize) -> Store {
        Store { index: HashMap::new(), order: VecDeque::new(), touched: HashSet::new(), next: 0, cap: cap.max(1), swept: 0 }
    }

    /// The live examples.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        (self.next as usize).min(self.cap)
    }

    /// Changes with every example added: a drawing made at another generation is out of date.
    pub fn generation(&self) -> u64 {
        self.next
    }

    fn oldest(&self) -> u64 {
        self.next.saturating_sub(self.cap as u64)
    }

    pub fn add(&mut self, e: Example) {
        self.index.entry(e.key()).or_default().v.push((e.r, e.bonded, self.next));
        self.order.push_back(e.key());
        self.touched.insert(e.key());
        self.next += 1;
        // no prepare() for a while (nothing stale to guess over): settle once an eighth is dead
        if self.order.len() > self.cap + (self.cap / 8).max(1) {
            self.prepare();
        }
    }

    /// Sweeps the dead examples, sorts what [`Store::add`] appended (R) and rebuilds the running
    /// counts of the keys that changed: before [`Store::predict`]. Nothing to do when nothing was
    /// added since.
    pub fn prepare(&mut self) {
        if self.touched.is_empty() {
            return;
        }
        let oldest = self.oldest();
        while self.swept < oldest {
            if let Some(k) = self.order.pop_front() {
                self.touched.insert(k);
            }
            self.swept += 1;
        }
        for k in std::mem::take(&mut self.touched) {
            let Some(slot) = self.index.get_mut(&k) else { continue };
            slot.v.retain(|x| x.2 >= oldest);
            if slot.v.is_empty() {
                self.index.remove(&k);
                continue;
            }
            // sorted before plus an appended tail: the stable sort takes the sorted run as it is
            slot.v.sort_by(|a, b| a.0.total_cmp(&b.0));
            slot.bonded_before.clear();
            slot.bonded_before.push(0);
            let mut n = 0u32;
            for x in &slot.v {
                n += x.1 as u32;
                slot.bonded_before.push(n);
            }
        }
    }

    /// The verdict for a pair with `key` at `r` (Bohr): [`MIN_AGREE`] or more live examples within
    /// [`MATCH_R`], all agreeing; else unknown. Call [`Store::prepare`] after adding. Two binary
    /// searches: every entry is live after `prepare`, and the running count gives the bonded ones.
    pub fn predict(&self, key: Key, r: f64) -> Verdict {
        debug_assert!(self.touched.is_empty(), "prepare() after add()");
        let Some(slot) = self.index.get(&key) else { return Verdict::Unknown };
        let lo = slot.v.partition_point(|x| x.0 < r - MATCH_R);
        let hi = slot.v.partition_point(|x| x.0 <= r + MATCH_R).max(lo);
        let yes = (slot.bonded_before[hi] - slot.bonded_before[lo]) as usize;
        let no = hi - lo - yes;
        match (yes, no) {
            (n, 0) if n >= MIN_AGREE => Verdict::Bonded,
            (0, n) if n >= MIN_AGREE => Verdict::NotBonded,
            _ => Verdict::Unknown,
        }
    }
}

// ---- the file ----

/// What [`open`] read.
#[derive(Debug, Default)]
pub struct Loaded {
    /// The newest examples, at most the cap, oldest first.
    pub examples: Vec<Example>,
    /// Examples the file held when read (before the cap).
    pub total: usize,
    /// Lines skipped (torn, damaged or not examples).
    pub skipped: usize,
    /// The file's header names another format: it was left alone (module doc).
    pub refused: Option<String>,
    /// It was compacted to the newest `cap`.
    pub compacted: bool,
}

/// Reads `path` (nothing when it does not exist): the newest `cap` examples.
pub fn load(path: &Path, cap: usize) -> io::Result<Loaded> {
    let text = match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Loaded::default()),
        Err(e) => return Err(e),
    };
    let mut out = Loaded::default();
    let mut all = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(v) = t.strip_prefix(HEADER_WORD).map(str::trim) {
            if v.parse::<u32>().ok() != Some(FORMAT_VERSION) {
                out.refused = Some(format!("bond experience: {} is in format {v:?}, not {FORMAT_VERSION} (a newer build?): left alone", path.display()));
                return Ok(Loaded { refused: out.refused, ..Loaded::default() });
            }
            continue;
        }
        match Example::parse(t) {
            Some(e) => all.push(e),
            None => out.skipped += 1,
        }
    }
    out.total = all.len();
    let from = all.len().saturating_sub(cap);
    out.examples = all.split_off(from);
    Ok(out)
}

/// [`load`], then the compaction when the file holds more than 2 × `cap` examples (module doc):
/// the newest `cap` to a temporary file renamed over it. A failed compaction leaves the file as it
/// was and is not an error.
pub fn open(path: &Path, cap: usize) -> io::Result<Loaded> {
    let mut l = load(path, cap)?;
    if l.refused.is_none() && l.total > 2 * cap {
        let tmp = path.with_extension(format!("txt.tmp-{}", std::process::id()));
        let mut body = format!("{HEADER_WORD} {FORMAT_VERSION}\n");
        for e in &l.examples {
            body += &e.line();
            body.push('\n');
        }
        let wrote = std::fs::File::create(&tmp).and_then(|mut f| {
            f.write_all(body.as_bytes())?;
            f.sync_all()
        });
        if wrote.and_then(|_| std::fs::rename(&tmp, path)).is_ok() {
            l.compacted = true;
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok(l)
}

/// The file does not end in a newline (a torn last line), or is empty.
fn ends_in_newline(path: &Path) -> io::Result<bool> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    if len == 0 {
        return Ok(true);
    }
    f.seek(SeekFrom::End(-1))?;
    let mut b = [0u8; 1];
    f.read_exact(&mut b)?;
    Ok(b[0] == b'\n')
}

/// Appends `ex` to `path` in one write (module doc), creating it with its header first.
pub fn append(path: &Path, ex: &[Example]) -> io::Result<()> {
    if ex.is_empty() {
        return Ok(());
    }
    // in append mode, so a header written after another writer's first batch (two writers creating
    // it at once) lands after that batch rather than over it; a reader takes a header anywhere
    match std::fs::OpenOptions::new().append(true).create_new(true).open(path) {
        Ok(mut f) => f.write_all(format!("{HEADER_WORD} {FORMAT_VERSION}\n").as_bytes())?,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let mut buf = String::with_capacity(ex.len() * 28 + 1);
    if !ends_in_newline(path)? {
        buf.push('\n');
    }
    for e in ex {
        buf += &e.line();
        buf.push('\n');
    }
    let mut f = std::fs::OpenOptions::new().append(true).open(path)?;
    f.write_all(buf.as_bytes())
}

// ---- the viewer's side ----

/// Where the examples live (module doc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    /// The session's memory, for this surface tag.
    Memory(String),
    /// A shared collection's folder.
    Collection(PathBuf),
}

/// Where `s`'s examples go: its shared collection's folder (the estimated solver learning into one
/// under `collections_dir`), else memory under the scene's surface tag.
pub fn place_for(collections_dir: Option<&Path>, s: &crate::sim::Snapshot) -> Place {
    if let (Some(dir), Some(sim::ModelSource::Collection { slug, .. })) = (collections_dir, s.model.as_ref().map(|m| &m.source)) {
        if !slug.is_empty() {
            return Place::Collection(dir.join(slug));
        }
    }
    let sc = &s.dynamics.scene;
    Place::Memory(format!("h={:?} sigma={:?} ionic={:?} charge={} mult={:?} engine={:?}", sc.cell.h, sc.sigma, sc.ionic, sc.charge, sc.multiplicity, s.engine))
}

/// How one drawn bond is styled (`overlay::BondLook`).
pub use crate::overlay::BondLook;

/// The bonds to draw while the graph is stale (module doc, "Display"): a graph for the drawing
/// only — never the panels' — with each bond's look.
#[derive(Clone, Debug)]
pub struct Shown {
    pub graph: BondGraph,
    pub looks: Vec<BondLook>,
    /// Pairs experience decided visibly: drawn from experience, or the last graph's bond hidden.
    pub experience: usize,
    /// The last graph's bonds kept stale (experience unknown).
    pub unsure: usize,
    /// Which snapshot it is for: its `seq` and its graph.
    seq: u64,
    of: Arc<BondGraph>,
    /// What it was made from besides the graph (the nuclei, their positions, the store's
    /// generation): the same again reuses it for the next snapshot.
    z: Vec<u8>,
    pos: Vec<[f64; 3]>,
    generation: u64,
}

impl Shown {
    /// Is it for this snapshot?
    pub fn is_for(&self, d: &sim::DynSnapshot) -> bool {
        d.seq == self.seq && d.bonds.as_ref().is_some_and(|g| Arc::ptr_eq(g, &self.of))
    }

    /// Was it made from `g` over these nuclei at `generation`? (Then it is the drawing for them.)
    fn made_from(&self, g: &Arc<BondGraph>, nuclei: &[sim::Nucleus], generation: u64) -> bool {
        Arc::ptr_eq(g, &self.of)
            && self.generation == generation
            && self.pos.len() == nuclei.len()
            && self.z.iter().zip(&self.pos).zip(nuclei).all(|((&z, p), n)| z == n.z && *p == n.pos)
    }

    /// The corner line (module doc): `None` when experience decided nothing (the plain stale line
    /// stands).
    pub fn label(&self) -> Option<String> {
        if self.experience == 0 {
            return None;
        }
        let s = if self.experience == 1 { "" } else { "s" };
        let mut t = format!("bonds from experience ({} pair{s}) · last DFT t = {:.2} fs", self.experience, units::au_time_to_fs(self.of.sim_time));
        if self.unsure > 0 {
            t += &format!(" · {} unsure (dashed)", self.unsure);
        }
        Some(t)
    }
}

/// A bond experience calls bonded where the last graph had none: a plain shared bond at the middle
/// of the drawn strength scale (`overlay::strength_fraction` ½: ρ_b the geometric mean of the
/// detector's floor and `overlay::RHO_STRONG`) — its strength is not known, and at the scale's
/// bottom it would be fainter than the stale bonds around it. Display only: never in a panel.
pub fn plain_bond(i: usize, j: usize) -> sim::Bond {
    let rho_bcp = (sim::bonds::RHO_FLOOR * crate::overlay::RHO_STRONG).sqrt();
    sim::Bond { i, j, rho_bcp, enhancement: f64::NAN, s: 0.5, route: sim::Route::Saddle, multi_centre: false, laplacian_bcp: -1.0, laplacian_noise: 0.0 }
}

/// The drawing of the stale `dft` graph for the nuclei `z` at `pos` under `store`'s verdicts
/// (module doc, "Display"). `None` when the graph is not for these nuclei.
pub fn compose(dft: &BondGraph, z: &[u8], pos: &[[f64; 3]], store: &Store) -> Option<(BondGraph, Vec<BondLook>, usize, usize)> {
    let n = pos.len();
    if dft.n_nuclei != n || z.len() != n {
        return None;
    }
    let counts = neighbour_counts(pos);
    let verdict = |i: usize, j: usize| {
        let r = dist(pos[i], pos[j]);
        if !(r > 0.0 && r < sim::bonds::R_MAX) {
            return Verdict::Unknown;
        }
        store.predict(pair_key(z[i], counts[i], z[j], counts[j]), r)
    };
    let mut graph = BondGraph { bonds: Vec::new(), components: dft.components.clone(), step: dft.step, sim_time: dft.sim_time, n_nuclei: n };
    let mut looks = Vec::new();
    let (mut experience, mut unsure) = (0, 0);
    let mut had = std::collections::HashSet::new();
    for b in &dft.bonds {
        if b.i >= n || b.j >= n {
            continue;
        }
        had.insert((b.i.min(b.j), b.i.max(b.j)));
        match verdict(b.i, b.j) {
            Verdict::Bonded => {
                graph.bonds.push(b.clone());
                looks.push(BondLook::Experience);
                experience += 1;
            }
            Verdict::NotBonded => experience += 1,
            Verdict::Unknown => {
                graph.bonds.push(b.clone());
                looks.push(BondLook::Stale);
                unsure += 1;
            }
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            if !had.contains(&(i, j)) && verdict(i, j) == Verdict::Bonded {
                graph.bonds.push(plain_bond(i, j));
                looks.push(BondLook::Experience);
                experience += 1;
            }
        }
    }
    Some((graph, looks, experience, unsure))
}

/// The viewer's experience (module doc): the store of its [`Place`], the recorder and the drawing
/// for the latest stale snapshot.
pub struct Experience {
    store: Store,
    place: Option<Place>,
    /// The collection file's load in flight, and what was recorded meanwhile.
    loading: Option<std::thread::JoinHandle<io::Result<Loaded>>>,
    queued: Vec<Example>,
    /// The file is in use (loaded, readable, writable).
    file_ok: bool,
    /// Examples in the file as far as this session knows (what it read plus what it appended).
    file_total: usize,
    /// The last graph recorded, its nuclei and positions.
    last: Option<(Arc<BondGraph>, Vec<u8>, Vec<[f64; 3]>)>,
    shown: Option<Shown>,
    /// Drawings made ([`compose`] runs), for the tests: none while the guess is not wanted, one per
    /// change of positions, graph or store.
    pub composed: u64,
    cap: usize,
    /// Why the file is not used, once (a refused or unreadable file, a failed append).
    pub note: Option<String>,
}

impl Default for Experience {
    fn default() -> Experience {
        Experience::with_cap(CAP)
    }
}

impl Experience {
    pub fn with_cap(cap: usize) -> Experience {
        Experience { store: Store::with_cap(cap), place: None, loading: None, queued: Vec::new(), file_ok: false, file_total: 0, last: None, shown: None, composed: 0, cap, note: None }
    }

    /// The live examples.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    fn file(&self) -> Option<PathBuf> {
        match &self.place {
            Some(Place::Collection(dir)) => Some(dir.join(FILE_NAME)),
            _ => None,
        }
    }

    /// Moves to `place`: a new place starts its own store (a collection's from its file) and
    /// forgets the last graph recorded (it went to the old place's store).
    fn switch(&mut self, place: Place) {
        if self.place.as_ref() == Some(&place) {
            return;
        }
        self.store = Store::with_cap(self.cap);
        self.queued.clear();
        self.file_ok = false;
        self.file_total = 0;
        self.loading = None;
        self.shown = None;
        self.last = None;
        if let Place::Collection(dir) = &place {
            let (path, cap) = (dir.join(FILE_NAME), self.cap);
            self.loading = std::thread::Builder::new().name("bond-experience".into()).spawn(move || open(&path, cap)).ok();
        }
        self.place = Some(place);
    }

    /// Takes the finished load: its examples, then what was recorded meanwhile (appended now).
    fn poll(&mut self, wait: bool) {
        if !self.loading.as_ref().is_some_and(|h| wait || h.is_finished()) {
            return;
        }
        let res = self.loading.take().map(|h| h.join().unwrap_or_else(|_| Err(io::Error::other("the load panicked"))));
        match res {
            Some(Ok(l)) if l.refused.is_none() => {
                l.examples.iter().for_each(|&e| self.store.add(e));
                self.file_total = if l.compacted { l.examples.len() } else { l.total };
                self.file_ok = true;
            }
            Some(Ok(l)) => self.note = l.refused,
            Some(Err(e)) => self.note = Some(format!("bond experience: the file could not be read ({e}): this session keeps it in memory")),
            None => {}
        }
        let q = std::mem::take(&mut self.queued);
        self.commit(q);
    }

    /// Adds `ex` to the store and, with the file in use, appends it.
    fn commit(&mut self, ex: Vec<Example>) {
        if ex.is_empty() {
            return;
        }
        ex.iter().for_each(|&e| self.store.add(e));
        let Some(path) = self.file().filter(|_| self.file_ok) else { return };
        match append(&path, &ex) {
            Ok(()) => {
                self.file_total += ex.len();
                if self.file_total > 2 * self.cap {
                    // compact off the frame (module doc); the store already holds the newest
                    let cap = self.cap;
                    let _ = std::thread::Builder::new().name("bond-experience-compact".into()).spawn(move || open(&path, cap));
                    self.file_total = self.cap;
                }
            }
            Err(e) => {
                self.file_ok = false;
                self.note = Some(format!("bond experience: could not append to {} ({e}): this session keeps it in memory", path.display()));
            }
        }
    }

    /// One snapshot (module doc): the place, the recording of a fresh graph, and, when `guess`
    /// (the viewer wants the drawing: the setting on, bonds shown, the estimated solver), the
    /// drawing over a stale one ([`Experience::refresh`]).
    pub fn on_snapshot(&mut self, d: &sim::DynSnapshot, place: Place, guess: bool) {
        self.switch(place);
        self.poll(false);
        let n = d.scene.nuclei.len();
        if let Some(g) = d.bonds.as_ref().filter(|g| g.n_nuclei == n && n >= 2 && !d.bonds_stale && !d.forces_stale) {
            let z: Vec<u8> = d.scene.nuclei.iter().map(|n| n.z).collect();
            let pos: Vec<[f64; 3]> = d.scene.nuclei.iter().map(|n| n.pos).collect();
            let seen = self.last.as_ref().is_some_and(|(lg, lz, lp)| Arc::ptr_eq(lg, g) || (*lz == z && *lp == pos));
            if !seen {
                let ex = examples(&z, &pos, g);
                self.last = Some((g.clone(), z, pos));
                if self.loading.is_some() {
                    self.queued.extend(ex);
                } else {
                    self.commit(ex);
                }
            }
        }
        self.refresh(d, guess);
    }

    /// The drawing for `d` (module doc, "Display"): only when `guess` and `d`'s graph is stale; else
    /// none, and nothing is computed. The last drawing is reused, re-stamped for `d`, when it was
    /// made from the same graph, nuclei, positions and store (a paused scene, stale snapshots after
    /// unrelated commands, every frame between snapshots). The viewer calls this per snapshot and
    /// per frame (the setting switched on while no snapshot comes).
    pub fn refresh(&mut self, d: &sim::DynSnapshot, guess: bool) {
        let n = d.scene.nuclei.len();
        let Some(g) = d.bonds.as_ref().filter(|g| guess && d.bonds_stale && g.n_nuclei == n && n >= 2) else {
            self.shown = None;
            return;
        };
        self.store.prepare();
        let generation = self.store.generation();
        if let Some(s) = self.shown.as_mut().filter(|s| s.made_from(g, &d.scene.nuclei, generation)) {
            s.seq = d.seq;
            return;
        }
        let z: Vec<u8> = d.scene.nuclei.iter().map(|n| n.z).collect();
        let pos: Vec<[f64; 3]> = d.scene.nuclei.iter().map(|n| n.pos).collect();
        self.composed += 1;
        self.shown = compose(g, &z, &pos, &self.store).map(|(graph, looks, experience, unsure)| Shown { graph, looks, experience, unsure, seq: d.seq, of: g.clone(), z, pos, generation });
    }

    /// The drawing for the latest stale snapshot, when there is one.
    pub fn shown(&self) -> Option<&Shown> {
        self.shown.as_ref()
    }

    /// Waits for a load in flight (tests).
    #[cfg(test)]
    pub fn wait_loaded(&mut self) {
        self.poll(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("atomic-sandbox-bond-exp-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn bond(i: usize, j: usize) -> sim::Bond {
        sim::Bond { i, j, rho_bcp: 0.2, enhancement: 1.4, s: 0.5, route: sim::Route::Saddle, multi_centre: false, laplacian_bcp: -1.0, laplacian_noise: 0.1 }
    }

    fn graph(n: usize, pairs: &[(usize, usize)]) -> Arc<BondGraph> {
        Arc::new(BondGraph { bonds: pairs.iter().map(|&(i, j)| bond(i, j)).collect(), components: vec![(0..n).collect()], step: 3, sim_time: 100.0, n_nuclei: n })
    }

    /// A snapshot of H nuclei at `pos` carrying `g`.
    fn snap(pos: &[[f64; 3]], g: Option<Arc<BondGraph>>, stale: bool, seq: u64) -> sim::DynSnapshot {
        let mut d = sim::DynSnapshot::default();
        d.scene.nuclei = pos.iter().map(|&p| sim::Nucleus::new(1, p)).collect();
        d.bonds = g;
        d.bonds_stale = stale;
        d.forces_stale = false;
        d.seq = seq;
        d
    }

    fn ex(za: u8, zb: u8, ca: u8, cb: u8, r: f64, bonded: bool) -> Example {
        Example { za, zb, ca, cb, r, bonded }
    }

    fn store_of(e: &[Example]) -> Store {
        let mut s = Store::default();
        e.iter().for_each(|&e| s.add(e));
        s.prepare();
        s
    }

    #[test]
    fn neighbour_counts_are_within_r_max_and_capped() {
        // a line of nuclei 3 Bohr apart: within 8 Bohr are the ones up to two away each side
        let pos: Vec<[f64; 3]> = (0..5).map(|k| [3.0 * k as f64, 0.0, 0.0]).collect();
        assert_eq!(neighbour_counts(&pos), vec![2, 3, 4, 3, 2]);
        // R_MAX is exclusive, like the detector's
        assert_eq!(neighbour_counts(&[[0.0; 3], [sim::bonds::R_MAX, 0.0, 0.0]]), vec![0, 0]);
        // a crowd: capped
        let crowd: Vec<[f64; 3]> = (0..10).map(|k| [0.5 * k as f64, 0.0, 0.0]).collect();
        assert!(neighbour_counts(&crowd).iter().all(|&c| c == COUNT_CAP));
        assert!(neighbour_counts(&[]).is_empty());
    }

    #[test]
    fn examples_cover_pairs_within_reach_with_sorted_ends() {
        // O at the origin, H at 1.8, a far H at 9 (out of the O's reach, within the near H's)
        let z = [8, 1, 1];
        let pos = [[0.0; 3], [1.8, 0.0, 0.0], [9.0, 0.0, 0.0]];
        let g = graph(3, &[(0, 1)]);
        let e = examples(&z, &pos, &g);
        assert_eq!(e.len(), 2, "O–H and H–H; O–far H is past R_MAX: {e:?}");
        // counts: O 1, near H 2, far H 1; the smaller (Z, count) end first
        assert_eq!(e[0], ex(1, 8, 2, 1, 1.8, true));
        assert_eq!((e[1].key(), e[1].bonded), ((1, 1, 1, 2), false));
        assert!((e[1].r - 7.2).abs() < 1e-12);
        assert_eq!(pair_key(1, 2, 8, 1), pair_key(8, 1, 1, 2), "either order");
        assert_eq!(pair_key(1, 3, 1, 2), (1, 1, 2, 3), "same Z: the counts sorted");
        assert!(examples(&z[..2], &pos, &g).is_empty(), "a count mismatch: nothing");
    }

    #[test]
    fn prediction_needs_three_agreeing_examples_of_the_same_key_within_the_window() {
        let k = (1, 1, 2, 2);
        let (b, nb) = (true, false);
        // all agree, bonded
        let s = store_of(&[ex(1, 1, 2, 2, 1.40, b), ex(1, 1, 2, 2, 1.45, b), ex(1, 1, 2, 2, 1.52, b)]);
        assert_eq!(s.predict(k, 1.46), Verdict::Bonded);
        // all agree, not bonded
        let s = store_of(&[ex(1, 1, 2, 2, 4.0, nb), ex(1, 1, 2, 2, 4.05, nb), ex(1, 1, 2, 2, 3.95, nb)]);
        assert_eq!(s.predict(k, 4.0), Verdict::NotBonded);
        // mixed: unknown
        let s = store_of(&[ex(1, 1, 2, 2, 2.0, b), ex(1, 1, 2, 2, 2.02, b), ex(1, 1, 2, 2, 2.04, b), ex(1, 1, 2, 2, 2.06, nb)]);
        assert_eq!(s.predict(k, 2.03), Verdict::Unknown);
        // fewer than three: unknown
        let s = store_of(&[ex(1, 1, 2, 2, 1.4, b), ex(1, 1, 2, 2, 1.41, b)]);
        assert_eq!(s.predict(k, 1.4), Verdict::Unknown);
        // the window is ±MATCH_R, inclusive
        let s = store_of(&[ex(1, 1, 2, 2, 1.30, b), ex(1, 1, 2, 2, 1.40, b), ex(1, 1, 2, 2, 1.50, b), ex(1, 1, 2, 2, 1.75, nb)]);
        assert_eq!(s.predict(k, 1.40), Verdict::Bonded);
        assert_eq!(s.predict(k, 1.55), Verdict::Unknown, "only 1.50 within 0.1");
        // other counts or another Z: no match
        let s = store_of(&[ex(1, 1, 2, 2, 1.4, b), ex(1, 1, 2, 2, 1.4, b), ex(1, 1, 2, 2, 1.4, b)]);
        assert_eq!(s.predict((1, 1, 1, 2), 1.4), Verdict::Unknown, "other counts");
        assert_eq!(s.predict((1, 8, 2, 2), 1.4), Verdict::Unknown, "another Z");
        // out-of-order adds are sorted by prepare()
        let s = store_of(&[ex(1, 1, 2, 2, 3.0, nb), ex(1, 1, 2, 2, 1.4, b), ex(1, 1, 2, 2, 1.42, b), ex(1, 1, 2, 2, 1.38, b)]);
        assert_eq!(s.predict(k, 1.4), Verdict::Bonded);
    }

    #[test]
    fn the_store_keeps_the_newest_cap() {
        let mut s = Store::with_cap(8);
        // eight old "bonded" ones, then eight new "not bonded" at the same R: the old ones age out
        (0..8).for_each(|_| s.add(ex(1, 1, 1, 1, 1.4, true)));
        s.prepare();
        assert_eq!(s.predict((1, 1, 1, 1), 1.4), Verdict::Bonded);
        (0..5).for_each(|_| s.add(ex(1, 1, 1, 1, 1.4, false)));
        s.prepare();
        assert_eq!(s.len(), 8);
        assert_eq!(s.predict((1, 1, 1, 1), 1.4), Verdict::Unknown, "3 old + 5 new live: mixed");
        (0..3).for_each(|_| s.add(ex(1, 1, 1, 1, 1.4, false)));
        s.prepare();
        assert_eq!(s.predict((1, 1, 1, 1), 1.4), Verdict::NotBonded, "only the newest 8 count");
        let held: usize = s.index.values().map(|slot| slot.v.len()).sum();
        assert!(held <= 8 + 8 / 8 + 1, "swept: {held}");
    }

    #[test]
    fn lines_round_trip_and_damage_is_refused() {
        let e = ex(1, 8, 2, 1, 1.8123456, true);
        let l = e.line();
        let back = Example::parse(&l).unwrap();
        assert_eq!((back.key(), back.bonded), (e.key(), true));
        assert!((back.r - 1.8123).abs() < 1e-9, "four decimals");
        // a torn line, a changed digit, a non-canonical order, a count over the cap
        assert!(Example::parse(&l[..l.len() - 3]).is_none());
        assert!(Example::parse(&l.replacen("1.8123", "1.8124", 1)).is_none());
        let swapped = { let b = "8 1 1 2 1.8000 1"; format!("{b} {:08x}", fnv1a(b)) };
        assert!(Example::parse(&swapped).is_none());
        let over = { let b = "1 1 7 7 1.8000 1"; format!("{b} {:08x}", fnv1a(b)) };
        assert!(Example::parse(&over).is_none());
    }

    #[test]
    fn the_file_round_trips_and_a_torn_last_line_is_ignored() {
        let d = tmp("roundtrip");
        let p = d.join(FILE_NAME);
        assert_eq!(load(&p, CAP).unwrap().total, 0, "no file: nothing");
        let a = [ex(1, 1, 1, 1, 1.4, true), ex(1, 8, 2, 1, 1.8, true)];
        append(&p, &a).unwrap();
        append(&p, &[ex(1, 1, 2, 2, 5.0, false)]).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.starts_with("bond-experience 1\n"), "{text}");
        assert_eq!(text.matches("bond-experience").count(), 1, "one header");
        let l = load(&p, CAP).unwrap();
        assert_eq!((l.total, l.skipped), (3, 0));
        assert_eq!(l.examples[2].key(), (1, 1, 2, 2));
        // a crash mid-line: the torn tail is skipped, and the next batch starts on its own line
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"1 1 1 1 1.40").unwrap();
        drop(f);
        let l = load(&p, CAP).unwrap();
        assert_eq!((l.total, l.skipped), (3, 1));
        append(&p, &[ex(1, 1, 1, 1, 1.5, true)]).unwrap();
        let l = load(&p, CAP).unwrap();
        assert_eq!((l.total, l.skipped), (4, 1), "the new line whole, the torn one still skipped");
        assert!((l.examples[3].r - 1.5).abs() < 1e-12);
        // a header of another version: left alone, nothing read
        let q = d.join("newer.txt");
        std::fs::write(&q, format!("bond-experience 2\n{}\n", a[0].line())).unwrap();
        let l = open(&q, CAP).unwrap();
        assert!(l.refused.is_some() && l.examples.is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_cap_keeps_the_newest_and_compaction_rewrites_past_twice_the_cap() {
        let d = tmp("compact");
        let p = d.join(FILE_NAME);
        let cap = 10;
        let all: Vec<Example> = (0..25).map(|k| ex(1, 1, 1, 1, 1.0 + 0.01 * k as f64, k % 2 == 0)).collect();
        append(&p, &all[..20]).unwrap();
        // 20 = 2 × cap: read the newest 10, not compacted
        let l = open(&p, cap).unwrap();
        assert_eq!((l.total, l.examples.len(), l.compacted), (20, 10, false));
        assert!((l.examples[0].r - 1.10).abs() < 1e-9 && (l.examples[9].r - 1.19).abs() < 1e-9, "the newest");
        // past 2 × cap: compacted to the newest 10 through a temporary file
        append(&p, &all[20..]).unwrap();
        let l = open(&p, cap).unwrap();
        assert_eq!((l.total, l.examples.len(), l.compacted), (25, 10, true));
        let l = load(&p, cap).unwrap();
        assert_eq!((l.total, l.skipped), (10, 0), "the file now holds the newest 10");
        assert!((l.examples[0].r - 1.15).abs() < 1e-9);
        assert!(std::fs::read_to_string(&p).unwrap().starts_with("bond-experience 1\n"));
        let leftovers: Vec<_> = std::fs::read_dir(&d).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name() != FILE_NAME).collect();
        assert!(leftovers.is_empty(), "no temporary file left: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn two_writers_interleave_whole_lines() {
        let d = tmp("writers");
        let p = d.join(FILE_NAME);
        let threads: Vec<_> = (0..2u8)
            .map(|w| {
                let p = p.clone();
                std::thread::spawn(move || {
                    for k in 0..150 {
                        let batch: Vec<Example> = (0..3).map(|m| ex(1, 1 + w, m, m, 1.0 + 0.001 * k as f64, w == 0)).collect();
                        append(&p, &batch).unwrap();
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        let l = load(&p, CAP).unwrap();
        assert_eq!((l.total, l.skipped), (900, 0), "every line whole");
        assert_eq!(l.examples.iter().filter(|e| e.zb == 1).count(), 450);
        // interleaved at line granularity by hand, with a torn line in the middle: the whole ones read
        let q = d.join("mixed.txt");
        let (a, b) = (ex(1, 1, 1, 1, 1.4, true), ex(1, 2, 1, 1, 2.0, false));
        std::fs::write(&q, format!("bond-experience 1\n{}\n{}\n1 1 1 1 1.4\n{}\nbond-experience 1\n{}\n", a.line(), b.line(), a.line(), b.line())).unwrap();
        let l = load(&q, CAP).unwrap();
        assert_eq!((l.total, l.skipped), (4, 1), "a second header (two creators) is fine");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Recording takes fresh graphs only, once each; a stale graph is never recorded.
    #[test]
    fn recording_takes_fresh_graphs_once() {
        let mut x = Experience::default();
        let place = Place::Memory("test".into());
        let pos = [[0.0; 3], [1.4, 0.0, 0.0]];
        let g = graph(2, &[(0, 1)]);
        x.on_snapshot(&snap(&pos, Some(g.clone()), true, 1), place.clone(), true);
        assert_eq!(x.len(), 0, "stale: not recorded");
        let mut drag = snap(&pos, Some(g.clone()), false, 2);
        drag.forces_stale = true;
        x.on_snapshot(&drag, place.clone(), true);
        assert_eq!(x.len(), 0, "forces stale (a drag before its solve): not recorded");
        x.on_snapshot(&snap(&pos, Some(g.clone()), false, 3), place.clone(), true);
        assert_eq!(x.len(), 1, "fresh: one pair");
        x.on_snapshot(&snap(&pos, Some(g.clone()), false, 4), place.clone(), true);
        assert_eq!(x.len(), 1, "the same graph again: not twice");
        x.on_snapshot(&snap(&pos, Some(graph(2, &[(0, 1)])), false, 5), place.clone(), true);
        assert_eq!(x.len(), 1, "a new graph at the very same geometry: not twice");
        let moved = [[0.0; 3], [1.5, 0.0, 0.0]];
        x.on_snapshot(&snap(&moved, Some(graph(2, &[(0, 1)])), false, 6), place.clone(), true);
        assert_eq!(x.len(), 2, "a new geometry's graph");
        x.on_snapshot(&snap(&moved[..1], Some(graph(2, &[(0, 1)])), false, 7), place.clone(), true);
        assert_eq!(x.len(), 2, "a graph of another nucleus count: never");
        // another place starts its own store
        x.on_snapshot(&snap(&pos, None, false, 8), Place::Memory("other".into()), true);
        assert_eq!(x.len(), 0);
    }

    /// A collection's examples are appended to its file and read back by the next session.
    #[test]
    fn a_collection_keeps_its_experience_in_its_folder() {
        let d = tmp("collection");
        let place = Place::Collection(d.clone());
        let mut x = Experience::default();
        for k in 0..4 {
            let pos = [[0.0; 3], [1.40 + 0.01 * k as f64, 0.0, 0.0]];
            x.on_snapshot(&snap(&pos, Some(graph(2, &[(0, 1)])), false, k), place.clone(), true);
            x.wait_loaded();
        }
        assert_eq!(x.len(), 4);
        assert_eq!(load(&d.join(FILE_NAME), CAP).unwrap().total, 4);
        let mut y = Experience::default();
        let pos = [[0.0; 3], [1.42, 0.0, 0.0]];
        let stale = snap(&pos, Some(graph(2, &[])), true, 9);
        y.on_snapshot(&stale, place.clone(), true);
        y.wait_loaded();
        assert_eq!(y.len(), 4, "read back");
        y.on_snapshot(&snap(&pos, Some(graph(2, &[])), true, 10), place, true);
        let shown = y.shown().expect("stale: a drawing");
        assert_eq!((shown.graph.bonds.len(), shown.looks.as_slice()), (1, &[BondLook::Experience][..]), "bonded from experience");
        assert!(y.note.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The drawing: experience-bonded pairs drawn from experience (the last graph's bond kept, or a
    /// plain one), experience-broken pairs dropped, unknown ones kept stale.
    #[test]
    fn compose_draws_experience_and_keeps_the_rest_stale() {
        // H₃ pulled apart: 0–1 at 1.6 (known bonded), 1–2 at 4.0 (known not bonded), 0–2 at 5.6 (unknown)
        let pos = [[0.0; 3], [1.6, 0.0, 0.0], [5.6, 0.0, 0.0]];
        let z = [1, 1, 1];
        let c = neighbour_counts(&pos);
        assert_eq!(c, vec![2, 2, 2]);
        let mut e = Vec::new();
        for r in [1.58, 1.6, 1.62] {
            e.push(ex(1, 1, 2, 2, r, true));
        }
        for r in [3.95, 4.0, 4.05] {
            e.push(ex(1, 1, 2, 2, r, false));
        }
        let s = store_of(&e);
        // the last DFT graph: the triangle's 1–2 and 0–2 bonded, 0–1 not
        let dft = graph(3, &[(1, 2), (0, 2)]);
        let (g, looks, experience, unsure) = compose(&dft, &z, &pos, &s).unwrap();
        let pairs: Vec<(usize, usize, BondLook)> = g.bonds.iter().zip(&looks).map(|(b, &l)| (b.i, b.j, l)).collect();
        assert_eq!(pairs, vec![(0, 2, BondLook::Stale), (0, 1, BondLook::Experience)], "1–2 broken, 0–2 unknown, 0–1 new");
        assert_eq!((experience, unsure), (2, 1));
        assert!((crate::overlay::strength_fraction(g.bonds[1].rho_bcp) - 0.5).abs() < 1e-6, "a new one at the middle of the strength scale");
        assert_eq!(g.bonds[1].kind(), sim::BondKind::Shared);
        assert_eq!(g.components, dft.components, "the molecules stay the DFT graph's");
        // nothing known: the stale graph as it was
        let (g, looks, experience, _) = compose(&dft, &z, &pos, &Store::default()).unwrap();
        assert_eq!((g.bonds, looks, experience), (dft.bonds.clone(), vec![BondLook::Stale; 2], 0));
        assert!(compose(&dft, &z[..2], &pos[..2], &s).is_none(), "another nucleus count");
    }

    #[test]
    fn the_label_names_the_experience_and_the_last_dft_time() {
        let pos = [[0.0; 3], [1.6, 0.0, 0.0]];
        let s = store_of(&[ex(1, 1, 1, 1, 1.6, true), ex(1, 1, 1, 1, 1.6, true), ex(1, 1, 1, 1, 1.6, true)]);
        let g = graph(2, &[]);
        let (graph_, looks, experience, unsure) = compose(&g, &[1, 1], &pos, &s).unwrap();
        let shown = Shown { graph: graph_, looks, experience, unsure, seq: 0, of: g.clone(), z: vec![1, 1], pos: pos.to_vec(), generation: s.generation() };
        let t = units::au_time_to_fs(100.0);
        assert_eq!(shown.label().unwrap(), format!("bonds from experience (1 pair) · last DFT t = {t:.2} fs"));
        let none = Shown { experience: 0, ..shown };
        assert!(none.label().is_none(), "nothing from experience: the plain stale line");
    }

    /// A small xorshift: the randomised tests' numbers, the same on every run.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn uniform(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The verdict as the first version counted it: every live example (the newest `cap`) of the
    /// key, one by one, within the window `[r − MATCH_R, r + MATCH_R]`.
    fn brute_force(all: &[Example], cap: usize, key: Key, r: f64) -> Verdict {
        let live = &all[all.len().saturating_sub(cap)..];
        let (mut yes, mut no) = (0usize, 0usize);
        for e in live.iter().filter(|e| e.key() == key && !(e.r < r - MATCH_R) && e.r <= r + MATCH_R) {
            if e.bonded {
                yes += 1;
            } else {
                no += 1;
            }
        }
        match (yes, no) {
            (n, 0) if n >= MIN_AGREE => Verdict::Bonded,
            (0, n) if n >= MIN_AGREE => Verdict::NotBonded,
            _ => Verdict::Unknown,
        }
    }

    /// Review I1: the running counts give exactly the verdicts of the one-by-one count, on a
    /// randomised store that ages examples out (a small cap, the settling in `add` included), at
    /// window edges (R on a 0.01 Bohr grid, so examples sit exactly at r ± MATCH_R) and with
    /// mixed, agreeing and too-few windows.
    #[test]
    fn the_lookup_matches_the_one_by_one_count_on_a_random_store() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let keys: [Key; 4] = [(1, 1, 2, 2), (1, 8, 2, 1), (1, 1, 1, 3), (8, 8, 6, 6)];
        let mut verdicts = [0usize; 3];
        for cap in [50usize, 333, 2000] {
            let mut store = Store::with_cap(cap);
            let mut all = Vec::new();
            for round in 0..40 {
                // a batch (a fresh graph), sometimes long enough to settle the store inside add()
                let batch = 1 + (rng.next() % if round % 7 == 3 { 3 * cap as u64 } else { 60 }) as usize;
                for _ in 0..batch {
                    let key = keys[(rng.next() % 4) as usize];
                    let r = (100.0 + (rng.next() % 300) as f64) / 100.0;
                    // bonded short, not bonded long, a noisy middle
                    let bonded = if r < 1.6 { rng.uniform() < 0.97 } else if r > 2.4 { rng.uniform() < 0.03 } else { rng.uniform() < 0.5 };
                    let e = Example { za: key.0, zb: key.1, ca: key.2, cb: key.3, r, bonded };
                    store.add(e);
                    all.push(e);
                }
                store.prepare();
                for _ in 0..200 {
                    let key = if rng.next() % 10 == 0 { (1, 6, 0, 0) } else { keys[(rng.next() % 4) as usize] };
                    // on the grid half the time (window edges), anywhere the rest
                    let r = if rng.next() % 2 == 0 { (90.0 + (rng.next() % 320) as f64) / 100.0 } else { 0.9 + 3.2 * rng.uniform() };
                    let want = brute_force(&all, cap, key, r);
                    assert_eq!(store.predict(key, r), want, "cap {cap}, round {round}, {key:?} at {r}");
                    verdicts[want as usize] += 1;
                }
            }
            let held: usize = store.index.values().map(|slot| slot.v.len()).sum();
            assert_eq!(held, cap.min(all.len()), "only the live ones held after prepare()");
        }
        assert!(verdicts.iter().all(|&n| n > 100), "every verdict exercised: {verdicts:?}");
    }

    /// Review I1: 300 atoms at liquid-water density (every count capped, so three keys share a full
    /// store of 200 000 examples between 3 and 6 Bohr, where MD keeps a liquid's neighbours —
    /// ≈ 4 400 in each lookup's window, agreeing away from 4.5 Bohr: the first version's window
    /// walk at its worst): a measurement, not a gate (wall-clock times flake under load), so it
    /// prints and asserts no time. Run it with `cargo test -p app --release --bin app --
    /// --ignored three_hundred_atoms --nocapture`. Measured 2026-11-13 (Ryzen 7
    /// 9800X3D): 2 907 pairs within R_MAX, compose 0.55 ms, its lookups 0.29 ms, the first
    /// version's window walk over the same pairs 2.6 ms (module doc, "Prediction").
    #[test]
    #[ignore = "timing measurement"]
    fn three_hundred_atoms_over_a_full_store_measured() {
        let mut rng = Rng(42);
        // a simple-cubic lattice with jitter at ≈ 0.0148 nuclei per Bohr³, one O per two H
        let (side, a) = (7usize, 4.07f64);
        let mut pos = Vec::new();
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    pos.push([x, y, z].map(|k| k as f64 * a + 0.4 * (rng.uniform() - 0.5)));
                }
            }
        }
        pos.truncate(300);
        let z: Vec<u8> = (0..pos.len()).map(|i| if i % 3 == 0 { 8 } else { 1 }).collect();
        assert!(neighbour_counts(&pos).iter().filter(|&&c| c == COUNT_CAP).count() > 250, "dense: counts capped");
        let keys = [pair_key(1, COUNT_CAP, 1, COUNT_CAP), pair_key(1, COUNT_CAP, 8, COUNT_CAP), pair_key(8, COUNT_CAP, 8, COUNT_CAP)];
        let mut store = Store::default();
        for k in 0..CAP {
            let key = keys[k % 3];
            let r = 3.0 + 3.0 * rng.uniform();
            store.add(Example { za: key.0, zb: key.1, ca: key.2, cb: key.3, r, bonded: r < 4.5 });
        }
        store.prepare();
        let dft = BondGraph { bonds: Vec::new(), components: vec![(0..pos.len()).collect()], step: 1, sim_time: 0.0, n_nuclei: pos.len() };
        let mut best = f64::INFINITY;
        let mut drawn = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let (g, ..) = compose(&dft, &z, &pos, &store).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            drawn = g.bonds.len();
        }
        assert!(drawn > 300, "the short pairs bonded from experience ({drawn})");
        // the lookups alone against the first version's window walk over the same pairs
        let counts = neighbour_counts(&pos);
        let mut asks = Vec::new();
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                let r = dist(pos[i], pos[j]);
                if r < sim::bonds::R_MAX {
                    asks.push((pair_key(z[i], counts[i], z[j], counts[j]), r));
                }
            }
        }
        let walk = |key: Key, r: f64| {
            let Some(slot) = store.index.get(&key) else { return 0usize };
            let lo = slot.v.partition_point(|x| x.0 < r - MATCH_R);
            slot.v[lo..].iter().take_while(|x| x.0 <= r + MATCH_R).filter(|x| x.1).count()
        };
        let time = |f: &dyn Fn()| {
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min)
        };
        let lookups = time(&|| asks.iter().for_each(|&(k, r)| {
            let _ = std::hint::black_box(store.predict(k, r));
        }));
        let walked = time(&|| asks.iter().for_each(|&(k, r)| {
            let _ = std::hint::black_box(walk(k, r));
        }));
        eprintln!("300 atoms ({} pairs within R_MAX), full store: compose {:.2} ms, its lookups {:.3} ms; the window walk {:.2} ms", asks.len(), best * 1e3, lookups * 1e3, walked * 1e3);
    }

    /// Review I1: nothing is composed while the guess is not wanted (the setting off, bonds hidden,
    /// another solver: the viewer passes `guess` false), and a drawing is reused while the graph,
    /// the nuclei and the store stay the same.
    #[test]
    fn the_guess_is_computed_only_when_wanted_and_only_when_something_changed() {
        let mut x = Experience::default();
        let place = Place::Memory("test".into());
        for k in 0..3 {
            let pos = [[0.0; 3], [1.40 + 0.01 * k as f64, 0.0, 0.0]];
            x.on_snapshot(&snap(&pos, Some(graph(2, &[(0, 1)])), false, k), place.clone(), false);
        }
        assert_eq!((x.len(), x.composed), (3, 0), "recorded whatever the display");
        let g = graph(2, &[]);
        let pos = [[0.0; 3], [1.41, 0.0, 0.0]];
        x.on_snapshot(&snap(&pos, Some(g.clone()), true, 10), place.clone(), false);
        assert_eq!(x.composed, 0, "not wanted: not computed");
        assert!(x.shown().is_none());
        x.on_snapshot(&snap(&pos, Some(g.clone()), true, 11), place.clone(), true);
        assert_eq!(x.composed, 1);
        assert_eq!(x.shown().unwrap().looks, vec![BondLook::Experience]);
        // the same graph and positions: reused, re-stamped for the new snapshot
        let again = snap(&pos, Some(g.clone()), true, 12);
        x.on_snapshot(&again, place.clone(), true);
        x.refresh(&again, true);
        assert_eq!(x.composed, 1, "nothing changed: reused");
        assert!(x.shown().unwrap().is_for(&again));
        // moved: made again
        let moved = [[0.0; 3], [1.43, 0.0, 0.0]];
        x.on_snapshot(&snap(&moved, Some(g.clone()), true, 13), place.clone(), true);
        assert_eq!(x.composed, 2, "moved");
        // the store grew (the file's load landing, say): made again over the same stale graph
        x.store.add(ex(1, 1, 1, 1, 3.0, false));
        x.on_snapshot(&snap(&moved, Some(g.clone()), true, 15), place.clone(), true);
        assert_eq!(x.composed, 3, "the store changed");
        // switched off: dropped, nothing made
        x.refresh(&snap(&moved, Some(g.clone()), true, 16), false);
        assert!(x.shown().is_none());
        assert_eq!(x.composed, 3);
    }

    /// Review Minor 5: a graph recorded at one place is recorded again at the next (the switch
    /// forgets it): a snapshot that arrives before the model names its collection does not keep
    /// the same graph out of the collection.
    #[test]
    fn a_new_place_records_the_last_graph_again() {
        let mut x = Experience::default();
        let pos = [[0.0; 3], [1.4, 0.0, 0.0]];
        let g = graph(2, &[(0, 1)]);
        x.on_snapshot(&snap(&pos, Some(g.clone()), false, 1), Place::Memory("before".into()), true);
        assert_eq!(x.len(), 1);
        x.on_snapshot(&snap(&pos, Some(g.clone()), false, 2), Place::Memory("after".into()), true);
        assert_eq!(x.len(), 1, "the same graph, recorded into the new place's store");
    }
}
