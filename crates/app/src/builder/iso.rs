//! A drawing's fragments and the exact atom mapping of one onto a database record (spec
//! `docs/superpowers/specs/2026-11-25-builder-database-design.md` §6.1–6.2).
//!
//! **The mapping rule.** A fragment matches a record only through a bijection of its atoms onto the
//! record's that keeps each atom's element and formal charge, keeps adjacency both ways (every
//! drawn bond is a record bond and vice versa), and keeps bond orders with one exception, the
//! Kekulé case: orders may differ on a bond that lies on a cycle (is not a bridge) in the drawn
//! fragment and in the record, provided every atom's sum of bond orders agrees. The per-atom sum is
//! required in every case; with all orders equal it agrees anyway, so this is the spec's rule.
//! Hydrogens are atoms like any other. A wrong mapping would put an atom at another element's
//! position, so anything not proven a match is [`MapResult::NoMatch`] or [`MapResult::CapHit`].
//!
//! **The search** is VF2-style backtracking. Each atom's key is (Z, formal charge, degree, sum of
//! bond orders), and a drawn atom's candidates are the record atoms with the same key. The drawn
//! atoms are visited in a connected order (the atom with the fewest candidates first, then always
//! an atom bonded to one already visited), so after the first atom the candidates are the unused
//! neighbours of a visited neighbour's image. Every candidate pair tried is one step; the search
//! gives up with [`MapResult::CapHit`] when the count reaches [`STEP_CAP`].
use super::records::{hill, Record};
use super::Drawing;
use std::collections::VecDeque;

/// The most candidate-pair tries one mapping may take before it gives up (spec §6.2).
pub const STEP_CAP: usize = 200_000;

/// The outcome of [`map_fragment`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MapResult {
    /// `m[i]` is the record atom of the fragment's `i`-th atom (`frag[i]`).
    Mapped(Vec<usize>),
    NoMatch,
    /// The search reached [`STEP_CAP`]: treated as not matched.
    CapHit,
}

/// The drawing's connected components (spec §6.1), lone atoms included: each sorted, in order of
/// their lowest atom index.
pub fn fragments(d: &Drawing) -> Vec<Vec<usize>> {
    d.components()
}

/// The fragment's lookup key, by the same rule as [`Record::formula_key`]: the Hill formula and the
/// net formal charge.
pub fn drawing_key(d: &Drawing, frag: &[usize]) -> (String, i32) {
    let z: Vec<u8> = frag.iter().map(|&i| d.atoms[i].z).collect();
    (hill(&z), frag.iter().map(|&i| d.atoms[i].charge as i32).sum())
}

/// Maps the drawn fragment `frag` (atom indices into `d`, e.g. one of [`fragments`]) onto `r` by
/// the module doc's rule. A `frag` with a bond to an atom outside it is not a closed fragment and
/// never matches.
pub fn map_fragment(d: &Drawing, frag: &[usize], r: &Record) -> MapResult {
    map_fragment_counting(d, frag, r).0
}

/// [`map_fragment`] and the number of steps it took.
pub(crate) fn map_fragment_counting(d: &Drawing, frag: &[usize], r: &Record) -> (MapResult, usize) {
    map_fragment_with_cap(d, frag, r, STEP_CAP)
}

fn map_fragment_with_cap(d: &Drawing, frag: &[usize], r: &Record, cap: usize) -> (MapResult, usize) {
    let n = frag.len();
    if n != r.z.len() || r.charge.len() != n {
        return (MapResult::NoMatch, 0);
    }
    // the drawn fragment in its own indices: position i is frag[i]
    let mut local = vec![None; d.atoms.len()];
    for (i, &a) in frag.iter().enumerate() {
        if local[a].replace(i).is_some() {
            return (MapResult::NoMatch, 0); // an atom listed twice
        }
    }
    let mut drawn_bonds = Vec::new();
    for b in &d.bonds {
        match (local[b.a], local[b.b]) {
            (Some(p), Some(q)) => drawn_bonds.push((p, q, b.order)),
            (None, None) => {}
            _ => return (MapResult::NoMatch, 0), // a bond leaves the fragment
        }
    }
    let z: Vec<u8> = frag.iter().map(|&a| d.atoms[a].z).collect();
    let charge: Vec<i8> = frag.iter().map(|&a| d.atoms[a].charge).collect();
    let (Some(g), Some(h)) = (Graph::new(&z, &charge, &drawn_bonds), Graph::new(&r.z, &r.charge, &r.bonds)) else {
        return (MapResult::NoMatch, 0);
    };
    if drawn_bonds.len() != r.bonds.len() {
        return (MapResult::NoMatch, 0);
    }
    let (mut gk, mut hk) = (g.key.clone(), h.key.clone());
    gk.sort_unstable();
    hk.sort_unstable();
    if gk != hk {
        return (MapResult::NoMatch, 0);
    }
    let mut s = Search::new(&g, &h, cap);
    let res = match s.extend(0) {
        Some(true) => MapResult::Mapped(s.m),
        Some(false) => MapResult::NoMatch,
        None => MapResult::CapHit,
    };
    (res, s.steps)
}

/// One side's molecular graph, atoms `0..n`.
struct Graph {
    /// Bond order between two atoms, 0 for none (n × n).
    order: Vec<Vec<u8>>,
    nbrs: Vec<Vec<usize>>,
    /// The bond between two atoms lies on a cycle (n × n; false without a bond).
    ring: Vec<Vec<bool>>,
    /// (Z, formal charge, degree, sum of bond orders).
    key: Vec<(u8, i8, usize, u32)>,
}

impl Graph {
    /// None for a bond out of range, to its own atom or listed twice.
    fn new(z: &[u8], charge: &[i8], bonds: &[(usize, usize, u8)]) -> Option<Graph> {
        let n = z.len();
        let mut order = vec![vec![0u8; n]; n];
        let mut nbrs = vec![Vec::new(); n];
        for &(a, b, o) in bonds {
            if a >= n || b >= n || a == b || o == 0 || order[a][b] != 0 {
                return None;
            }
            order[a][b] = o;
            order[b][a] = o;
            nbrs[a].push(b);
            nbrs[b].push(a);
        }
        let mut ring = vec![vec![false; n]; n];
        for &(a, b, _) in bonds {
            let on_cycle = reachable_without(&nbrs, a, b);
            ring[a][b] = on_cycle;
            ring[b][a] = on_cycle;
        }
        let key = (0..n).map(|i| (z[i], charge[i], nbrs[i].len(), nbrs[i].iter().map(|&j| order[i][j] as u32).sum())).collect();
        Some(Graph { order, nbrs, ring, key })
    }
}

/// `b` is reachable from its neighbour `a` without the bond a–b: the bond is not a bridge.
fn reachable_without(nbrs: &[Vec<usize>], a: usize, b: usize) -> bool {
    let mut seen = vec![false; nbrs.len()];
    seen[a] = true;
    let mut queue = VecDeque::from([a]);
    while let Some(u) = queue.pop_front() {
        for &v in &nbrs[u] {
            if (u, v) == (a, b) || seen[v] {
                continue;
            }
            if v == b {
                return true;
            }
            seen[v] = true;
            queue.push_back(v);
        }
    }
    false
}

/// The backtracking state: drawn graph `g`, record graph `h`.
struct Search<'a> {
    g: &'a Graph,
    h: &'a Graph,
    /// The drawn atoms in visiting order.
    order: Vec<usize>,
    /// A drawn neighbour visited before the atom, if any (its image's neighbours are the candidates).
    parent: Vec<Option<usize>>,
    /// The record atom of each drawn atom (`usize::MAX`: not yet mapped).
    m: Vec<usize>,
    used: Vec<bool>,
    steps: usize,
    cap: usize,
}

impl<'a> Search<'a> {
    fn new(g: &'a Graph, h: &'a Graph, cap: usize) -> Search<'a> {
        let n = g.key.len();
        let cands: Vec<usize> = (0..n).map(|i| h.key.iter().filter(|&&k| k == g.key[i]).count()).collect();
        let mut order = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];
        while order.len() < n {
            // the unvisited atom with the most visited neighbours, then the fewest candidates, then
            // the lowest index (a fresh start, for a disconnected fragment, has no visited neighbour)
            let next = (0..n)
                .filter(|&i| !visited[i])
                .min_by_key(|&i| (std::cmp::Reverse(g.nbrs[i].iter().filter(|&&j| visited[j]).count()), cands[i], i))
                .unwrap();
            parent[next] = order.iter().copied().find(|&j| g.order[next][j] != 0);
            visited[next] = true;
            order.push(next);
        }
        Search { g, h, order, parent, m: vec![usize::MAX; n], used: vec![false; n], steps: 0, cap }
    }

    /// Maps the drawn atoms `order[k..]`: Some(true) when the whole mapping is found, Some(false)
    /// when none extends the current one, None when the step count reached the cap.
    fn extend(&mut self, k: usize) -> Option<bool> {
        let Some(&i) = self.order.get(k) else { return Some(true) };
        let cands: Vec<usize> = match self.parent[i] {
            Some(p) => self.h.nbrs[self.m[p]].clone(),
            None => (0..self.h.key.len()).collect(),
        };
        for r in cands {
            if self.used[r] || self.h.key[r] != self.g.key[i] {
                continue;
            }
            self.steps += 1;
            if self.steps >= self.cap {
                return None;
            }
            if !self.feasible(k, i, r) {
                continue;
            }
            self.m[i] = r;
            self.used[r] = true;
            match self.extend(k + 1) {
                Some(false) => {}
                done => return done,
            }
            self.m[i] = usize::MAX;
            self.used[r] = false;
        }
        Some(false)
    }

    /// Drawn atom `i` (visited `k`-th) onto record atom `r` agrees with every atom mapped so far:
    /// bonded alike, and with the same order unless the bond is on a cycle on both sides (the
    /// Kekulé case; the keys already hold each atom's order sum equal).
    fn feasible(&self, k: usize, i: usize, r: usize) -> bool {
        self.order[..k].iter().all(|&j| {
            let (s, t) = (self.m[j], self.g.order[i][j]);
            let u = self.h.order[r][s];
            (t == 0) == (u == 0) && (t == u || (self.g.ring[i][j] && self.h.ring[r][s]))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::records::{Library, Source};

    fn rec(name: &str) -> &'static Record {
        Library::shipped().records().iter().find(|r| r.name == name).unwrap_or_else(|| panic!("no {name}"))
    }

    /// A graph-only record (no shape) for the cases the seed file has no entry for.
    fn graph_record(z: &[u8], bonds: &[(usize, usize, u8)]) -> Record {
        Record {
            name: "test".into(),
            synonyms: vec![],
            source: Source::PubChem { cid: 0 },
            z: z.to_vec(),
            charge: vec![0; z.len()],
            bonds: bonds.to_vec(),
            xyz: vec![[0.0; 3]; z.len()],
            xy: vec![[0.0; 2]; z.len()],
            stereo_defined: false,
        }
    }

    /// Draws a molecule into `d`: canonical atom `c` becomes drawing atom `first + perm[c]` (`perm`
    /// is a permutation of `0..z.len()`), each bond drawn with `bond` called `order` times.
    fn draw_into(d: &mut Drawing, z: &[u8], bonds: &[(usize, usize, u8)], perm: &[usize]) {
        let first = d.atoms.len();
        let mut inv = vec![0; z.len()];
        for (c, &k) in perm.iter().enumerate() {
            inv[k] = c;
        }
        for (k, &c) in inv.iter().enumerate() {
            d.add_atom(z[c], [k as f32, 0.0]);
        }
        for &(a, b, order) in bonds {
            for _ in 0..order {
                d.bond(first + perm[a], first + perm[b]).unwrap();
            }
        }
    }

    fn draw(z: &[u8], bonds: &[(usize, usize, u8)]) -> Drawing {
        let mut d = Drawing::default();
        draw_into(&mut d, z, bonds, &(0..z.len()).collect::<Vec<_>>());
        d
    }

    fn all(d: &Drawing) -> Vec<usize> {
        (0..d.atoms.len()).collect()
    }

    // ethanol as drawn: O, C(OH), C(H3), the O's H, two Hs on the first C, three on the second
    const ETHANOL_Z: [u8; 9] = [8, 6, 6, 1, 1, 1, 1, 1, 1];
    const ETHANOL_BONDS: [(usize, usize, u8); 8] = [(0, 1, 1), (1, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1), (2, 6, 1), (2, 7, 1), (2, 8, 1)];

    /// Benzene with its doubles at (0,1), (2,3), (4,5) (`shift` false) or (1,2), (3,4), (5,0).
    fn benzene(shift: bool) -> Drawing {
        let mut z = vec![6u8; 6];
        z.extend([1u8; 6]);
        let mut bonds = Vec::new();
        for i in 0..6 {
            let double = (i % 2 == 0) != shift;
            bonds.push((i, (i + 1) % 6, if double { 2 } else { 1 }));
            bonds.push((i, i + 6, 1));
        }
        draw(&z, &bonds)
    }

    /// `m` is a bijection onto `r`'s atoms that keeps Z, charge and every bond (orders up to the
    /// Kekulé rule, which the callers check where it matters).
    fn assert_valid(d: &Drawing, frag: &[usize], r: &Record, m: &[usize]) {
        assert_eq!(m.len(), r.z.len());
        let mut sorted = m.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..r.z.len()).collect::<Vec<_>>(), "not a bijection: {m:?}");
        for (i, &a) in frag.iter().enumerate() {
            assert_eq!(d.atoms[a].z, r.z[m[i]], "Z of fragment atom {i}");
            assert_eq!(d.atoms[a].charge, r.charge[m[i]], "charge of fragment atom {i}");
        }
        let pos = |a: usize| frag.iter().position(|&x| x == a).unwrap();
        let record_bond = |p: usize, q: usize| r.bonds.iter().find(|&&(a, b, _)| (a, b) == (p, q) || (a, b) == (q, p)).map(|b| b.2);
        let mut n = 0;
        for b in d.bonds.iter().filter(|b| frag.contains(&b.a)) {
            assert!(record_bond(m[pos(b.a)], m[pos(b.b)]).is_some(), "drawn bond {}–{} is not a record bond", b.a, b.b);
            n += 1;
        }
        assert_eq!(n, r.bonds.len());
    }

    #[test]
    fn ethanol_maps_in_any_atom_order() {
        let r = rec("ethanol");
        let perms: [[usize; 9]; 3] = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1, 0], [4, 0, 7, 2, 8, 1, 5, 3, 6]];
        for (t, perm) in perms.iter().enumerate() {
            let mut d = Drawing::default();
            if t == 2 {
                // a lone Na first, so fragment positions and drawing indices differ by one
                d.add_atom(11, [-5.0, 0.0]);
            }
            draw_into(&mut d, &ETHANOL_Z, &ETHANOL_BONDS, perm);
            let frag = fragments(&d).into_iter().find(|f| f.len() == 9).unwrap();
            match map_fragment(&d, &frag, r) {
                MapResult::Mapped(m) => {
                    assert_valid(&d, &frag, r, &m);
                    // no ring, so every order matches exactly
                    for b in &d.bonds {
                        let (p, q) = (frag.iter().position(|&x| x == b.a).unwrap(), frag.iter().position(|&x| x == b.b).unwrap());
                        let o = r.bonds.iter().find(|&&(a, c, _)| (a, c) == (m[p], m[q]) || (a, c) == (m[q], m[p])).unwrap().2;
                        assert_eq!(o, b.order);
                    }
                }
                other => panic!("permutation {t}: {other:?}"),
            }
        }
    }

    #[test]
    fn benzene_matches_in_either_kekule_form() {
        let r = rec("benzene");
        for shift in [false, true] {
            let d = benzene(shift);
            match map_fragment(&d, &all(&d), r) {
                MapResult::Mapped(m) => assert_valid(&d, &all(&d), r, &m),
                other => panic!("shift {shift}: {other:?}"),
            }
        }
    }

    #[test]
    fn only_ring_bonds_may_differ() {
        // two triangles joined by a bridge; the second form moves two order units onto the bridge
        // and keeps every atom's bond-order sum, so only the ring rule tells them apart
        let z = [6u8; 6];
        let form = |bridge: u8, side: u8, far: u8| {
            vec![(0, 1, side), (0, 2, side), (1, 2, far), (3, 4, side), (3, 5, side), (4, 5, far), (0, 3, bridge)]
        };
        let (one, three) = (form(1, 2, 1), form(3, 1, 2));
        let d = draw(&z, &one);
        assert!(matches!(map_fragment(&d, &all(&d), &graph_record(&z, &one)), MapResult::Mapped(_)));
        assert_eq!(map_fragment(&d, &all(&d), &graph_record(&z, &three)), MapResult::NoMatch);
        // the other way round too
        let d = draw(&z, &three);
        assert_eq!(map_fragment(&d, &all(&d), &graph_record(&z, &one)), MapResult::NoMatch);
        // ring bonds that differ but leave an atom's sum changed don't map either
        let unbalanced = vec![(0, 1, 2), (0, 2, 1), (1, 2, 1), (3, 4, 1), (3, 5, 1), (4, 5, 2), (0, 3, 3)];
        assert_eq!(map_fragment(&d, &all(&d), &graph_record(&z, &unbalanced)), MapResult::NoMatch);
    }

    #[test]
    fn near_misses_do_not_match() {
        let water = rec("water");
        let mut d = Drawing::default();
        let o = d.add_atom(8, [0.0, 0.0]);
        let h = d.add_atom(1, [1.0, 0.0]);
        d.bond(o, h);
        assert_eq!(map_fragment(&d, &all(&d), water), MapResult::NoMatch, "water missing one H");

        let ethanol = rec("ethanol");
        let mut z = ETHANOL_Z;
        z[0] = 16;
        let d = draw(&z, &ETHANOL_BONDS);
        assert_eq!(map_fragment(&d, &all(&d), ethanol), MapResult::NoMatch, "O → S");

        let mut d = draw(&ETHANOL_Z, &ETHANOL_BONDS);
        d.bond(0, 2);
        assert_eq!(map_fragment(&d, &all(&d), ethanol), MapResult::NoMatch, "an extra C–O ring bond");

        // a fragment list that leaves out a bonded atom is not a closed fragment
        let mut d = draw(&[8, 1, 1], &[(0, 1, 1), (0, 2, 1)]);
        let h3 = d.add_atom(1, [3.0, 0.0]);
        d.bond(0, h3);
        assert_eq!(map_fragment(&d, &[0, 1, 2], water), MapResult::NoMatch, "a bond leaves the fragment");
    }

    #[test]
    fn charges_must_agree() {
        let nh4 = rec("ammonium");
        let mut d = draw(&[7, 1, 1, 1, 1], &[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)]);
        assert_eq!(map_fragment(&d, &all(&d), nh4), MapResult::NoMatch, "neutral NH₄");
        d.cycle_charge(0);
        assert_eq!(d.atoms[0].charge, 1);
        match map_fragment(&d, &all(&d), nh4) {
            MapResult::Mapped(m) => assert_valid(&d, &all(&d), nh4, &m),
            other => panic!("NH₄⁺: {other:?}"),
        }
    }

    #[test]
    fn symmetric_molecules_match_inside_the_cap() {
        let methane = draw(&[6, 1, 1, 1, 1], &[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)]);
        let benzene = benzene(true);
        for (name, d) in [("methane", &methane), ("benzene", &benzene)] {
            let (res, steps) = map_fragment_counting(d, &all(d), rec(name));
            eprintln!("{name}: {steps} steps");
            assert!(matches!(res, MapResult::Mapped(_)), "{name}: {res:?}");
            assert!(steps < STEP_CAP / 100, "{name}: {steps} steps");
        }
    }

    #[test]
    fn the_cap_returns_cap_hit() {
        let d = benzene(false);
        assert_eq!(map_fragment_with_cap(&d, &all(&d), rec("benzene"), 1).0, MapResult::CapHit);
    }

    #[test]
    fn fragments_split_components() {
        // water, a lone Na and a second water, interleaved
        let mut d = Drawing::default();
        for z in [8, 11, 1, 8, 1, 1, 1] {
            d.add_atom(z, [0.0, 0.0]);
        }
        for (a, b) in [(0, 2), (3, 5), (0, 4), (3, 6)] {
            d.bond(a, b);
        }
        let f = fragments(&d);
        assert_eq!(f, vec![vec![0, 2, 4], vec![1], vec![3, 5, 6]]);
        assert_eq!(drawing_key(&d, &f[0]), ("H2O".into(), 0));
        assert_eq!(drawing_key(&d, &f[1]), ("Na".into(), 0));
    }

    #[test]
    fn drawing_keys_agree_with_record_keys() {
        let d = draw(&ETHANOL_Z, &ETHANOL_BONDS);
        assert_eq!(drawing_key(&d, &all(&d)), rec("ethanol").formula_key());
        let mut d = draw(&[7, 1, 1, 1, 1], &[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)]);
        d.cycle_charge(0);
        assert_eq!(drawing_key(&d, &all(&d)), rec("ammonium").formula_key());
    }
}
