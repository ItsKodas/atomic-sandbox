//! Bonds guessed from experience: the pure part (moved here unchanged from the app's
//! `bond_experience` module for the live-orbitals plan's Task 9, so the validate suite can score it
//! — V-A4, spec `docs/superpowers/specs/2026-11-14-live-orbitals-design.md` §7). The app keeps the
//! file I/O, where the examples live (`Place`), the drawing (`compose`) and the viewer's
//! `Experience`, and re-exports this module; its module doc is the narrative (user request
//! 2026-11-13).
//!
//! **Rule 1.** Nothing here is typed in per element: no radii, no valences, no bond lengths. Every
//! example is a verdict of the app's own density detector (`crate::bonds`) on the app's own DFT
//! density; the element enters only as the atomic number Z of the pair, the same way it enters the
//! solver.
//!
//! # Examples ([`examples`])
//! Every pair i < j of a fresh graph closer than `crate::bonds::R_MAX` becomes one [`Example`]: its
//! (Z, neighbour count) ends sorted together, R (Bohr), and whether the graph holds the pair. An
//! atom's neighbour count is the number of other nuclei within `R_MAX` (the detector's own reach),
//! capped at [`COUNT_CAP`]. The file line of an example ([`Example::line`], [`Example::parse`]):
//! `za zb ca cb R bonded check` (`check`: FNV-1a of the text before it, 8 hex digits).
//!
//! # Prediction ([`Store::predict`])
//! For a pair with the same (Z, count) ends, its examples with |R − R_ex| ≤ [`MATCH_R`] are counted
//! (the store is indexed by that key with R sorted and a running count of the bonded ones, so a
//! lookup is two binary searches). At least [`MIN_AGREE`] of them, all agreeing: that verdict; else
//! unknown. At most [`CAP`] examples are live (the newest). `MATCH_R` = 0.1 Bohr and `MIN_AGREE` = 3
//! are **display constants, not physics**: how close and how many before the view trusts what it
//! saw.
use crate::BondGraph;
use std::collections::{HashMap, HashSet, VecDeque};

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
        (canonical && e.za > 0 && e.ca <= COUNT_CAP && e.cb <= COUNT_CAP && e.r.is_finite() && e.r > 0.0 && e.r < crate::bonds::R_MAX).then_some(e)
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Each nucleus's count of other nuclei closer than `crate::bonds::R_MAX`, capped at [`COUNT_CAP`].
pub fn neighbour_counts(pos: &[[f64; 3]]) -> Vec<u8> {
    (0..pos.len()).map(|i| (0..pos.len()).filter(|&j| j != i && dist(pos[i], pos[j]) < crate::bonds::R_MAX).count().min(COUNT_CAP as usize) as u8).collect()
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
            if !(r > 0.0 && r < crate::bonds::R_MAX) {
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

    /// The live examples (public since the move out of the app: its tests and the validate suite
    /// count them).
    pub fn len(&self) -> usize {
        (self.next as usize).min(self.cap)
    }

    /// No live example.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(za: u8, zb: u8, ca: u8, cb: u8, r: f64, bonded: bool) -> Example {
        Example { za, zb, ca, cb, r, bonded }
    }

    fn bond(i: usize, j: usize) -> crate::Bond {
        crate::Bond { i, j, rho_bcp: 0.2, enhancement: 1.4, s: 0.5, route: crate::Route::Saddle, multi_centre: false, laplacian_bcp: -1.0, laplacian_noise: 0.1 }
    }

    fn graph(n: usize, pairs: &[(usize, usize)]) -> BondGraph {
        BondGraph { bonds: pairs.iter().map(|&(i, j)| bond(i, j)).collect(), components: vec![(0..n).collect()], step: 3, sim_time: 100.0, n_nuclei: n }
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
        assert_eq!(neighbour_counts(&[[0.0; 3], [crate::bonds::R_MAX, 0.0, 0.0]]), vec![0, 0]);
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

    /// Review I1, the lookup half of the app's `three_hundred_atoms_over_a_full_store_measured`
    /// (split at the move: the drawing, `compose`, stayed in the app; the store's internals came
    /// here): 300 atoms at liquid-water density over a full store of [`CAP`] examples, the
    /// lookups against the first version's walk over each window — the same lattice, store and
    /// pairs as the app's half (the same seeds). A measurement, not a gate: prints, asserts no
    /// time. `cargo test -p sim --release -- --ignored three_hundred_atoms --nocapture`. Measured
    /// 2026-11-13 (Ryzen 7 9800X3D, before the split): 2 907 pairs within R_MAX, the lookups
    /// 0.29 ms, the window walk 2.6 ms.
    #[test]
    #[ignore = "timing measurement"]
    fn three_hundred_atoms_lookups_over_a_full_store_measured() {
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
        let counts = neighbour_counts(&pos);
        let mut asks = Vec::new();
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                let r = dist(pos[i], pos[j]);
                if r < crate::bonds::R_MAX {
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
        eprintln!("300 atoms ({} pairs within R_MAX), full store: the lookups {:.3} ms; the window walk {:.2} ms", asks.len(), lookups * 1e3, walked * 1e3);
    }
}
