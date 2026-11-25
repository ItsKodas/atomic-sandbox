//! The hand-off geometry and its labels (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md`
//! §6.1, §4 Goal 5 and §8): where Insert / Replace put the drawn atoms, and the footer's line saying
//! where that shape comes from.
//!
//! **The layout (§6.1).** [`embed::embed`] lays out the whole drawing once, exactly as before the
//! database existed. Each fragment of two or more atoms is then looked up through a
//! [`RecordSource`]; a found fragment's atoms take its record's 3D shape (Å → Bohr) through the
//! atom mapping, centred on the centroid the embedding gave that fragment. Only the matched
//! fragments then move, pushed apart along the line between fragment centroids until every pair of
//! atoms in different fragments is at least [`FRAGMENT_GAP`] apart, and the whole is centred last as
//! the embedding is. A matched fragment the push cannot clear within [`MAX_PUSH_PASSES`] (trapped
//! between atoms that do not move) goes back to the embedding's positions, labelled approximate
//! ([`Why::NoRoom`]), and the others are pushed again. Lone atoms and fragments not found keep the
//! embedding's positions; a drawing with no matched fragment is handed off bit for bit as the
//! embedding gives it.
//!
//! **Answers are checked** ([`valid_mapping`]): a source's mapping must keep every element and
//! formal charge and take every drawn bond to a record bond, or the fragment is approximate.
//!
//! **Rule 1 (spec §3).** A record only places nuclei here; nothing from it leaves the builder.
use super::embed::{self, closest};
use super::iso;
use super::records::{Library, Record, Source};
use super::Drawing;

type V3 = [f64; 3];

/// The least distance between atoms of different fragments after the push (Bohr): the gap Insert
/// keeps from the scene's atoms ([`embed::BESIDE_GAP`]).
pub const FRAGMENT_GAP: f64 = 3.0;

/// Added to each push beyond the shortfall (Bohr), so a push that is not along the closest pair's
/// line still converges in a few passes.
const PUSH_SLACK: f64 = 0.05;

/// The most passes over the fragment pairs the push takes. A matched fragment still closer than
/// [`FRAGMENT_GAP`] to another fragment after them goes back to the embedding ([`Why::NoRoom`]).
pub const MAX_PUSH_PASSES: usize = 1000;

/// Why a fragment's shape is the embedding's guess.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Why {
    /// No record maps onto it (shipped list, cache and PubChem asked).
    NotInDatabase,
    /// PubChem was not asked or did not answer: the online setting is off, the process has no
    /// network (`--replay`, `--screenshot`, tests), or the lookup timed out or failed.
    Offline,
    /// The online lookup is in flight (or waiting out [`super::online::DEBOUNCE`]).
    LookingUp,
    /// A record maps onto it, but the push apart could not clear its shape of the other fragments'
    /// atoms within [`MAX_PUSH_PASSES`] (it is trapped between atoms that do not move), so it keeps
    /// the embedding's positions rather than overlap them.
    NoRoom,
}

/// Where one fragment's hand-off shape comes from (the footer's label).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentSource {
    /// A PubChem record's conformer; `stereo_note` when the record is one stereoisomer of a graph
    /// that has several, so the drawing did not decide it (§6.4).
    PubChem { name: String, cid: u64, stereo_note: bool },
    /// One of our own LIRA-relaxed entries (§5.1).
    Lira,
    /// The embedding's guess.
    Approximate(Why),
}

/// The hand-off geometry: the drawing's atoms in Bohr (drawing order, centred as
/// [`embed::centre`] does) and each looked-up fragment's source (fragments of two or more atoms, in
/// [`iso::fragments`] order; lone atoms have no shape to label).
#[derive(Clone, Debug, PartialEq)]
pub struct Shaped {
    pub positions: Vec<V3>,
    pub sources: Vec<(Vec<usize>, FragmentSource)>,
}

/// A record source's answer for one fragment.
#[derive(Clone, Debug, PartialEq)]
pub enum Lookup {
    /// A record and the mapping onto it: `m[i]` is the record atom of the fragment's `i`-th atom.
    Found(Record, Vec<usize>),
    NotFound,
    /// Being looked up online.
    Pending,
    /// The online lookup failed, or was not allowed.
    Offline,
}

/// Where fragments are looked up (spec §6.3).
pub trait RecordSource {
    fn find(&self, d: &Drawing, frag: &[usize]) -> Lookup;
}

/// The shipped list only ([`super::online::Layered`] adds the cache and PubChem): the first
/// candidate with the fragment's formula and charge that maps exactly.
pub struct LocalOnly<'a>(pub &'a Library);

impl RecordSource for LocalOnly<'_> {
    fn find(&self, d: &Drawing, frag: &[usize]) -> Lookup {
        for r in self.0.candidates(&iso::drawing_key(d, frag)) {
            if let iso::MapResult::Mapped(m) = iso::map_fragment(d, frag, r) {
                return Lookup::Found(r.clone(), m);
            }
        }
        Lookup::NotFound
    }
}

/// The hand-off geometry of `d` (the module doc's layout).
pub fn shape(d: &Drawing, src: &dyn RecordSource) -> Shaped {
    let (mut positions, sources, matched) = place(d, src);
    if matched {
        embed::centre(&mut positions);
    }
    Shaped { positions, sources }
}

/// [`shape`] before the final centring, and whether any fragment matched (only then is it centred).
fn place(d: &Drawing, src: &dyn RecordSource) -> (Vec<V3>, Vec<(Vec<usize>, FragmentSource)>, bool) {
    place_from(d, src, embed::embed(d))
}

/// [`place`] from the whole drawing's embedding `x0` (the tests give one by hand).
fn place_from(d: &Drawing, src: &dyn RecordSource, x0: Vec<V3>) -> (Vec<V3>, Vec<(Vec<usize>, FragmentSource)>, bool) {
    let mut x = x0.clone();
    let mut sources = Vec::new();
    let mut groups = Vec::new(); // (fragment, matched)
    for frag in iso::fragments(d) {
        if frag.len() < 2 {
            groups.push((frag, false));
            continue;
        }
        let (source, matched) = match src.find(d, &frag) {
            Lookup::Found(r, m) if valid_mapping(d, &frag, &r, &m) => {
                put_record(&mut x, &frag, &r, &m);
                let source = match r.source {
                    Source::PubChem { cid } => FragmentSource::PubChem { name: r.name.clone(), cid, stereo_note: r.stereo_defined },
                    Source::Lira { .. } => FragmentSource::Lira,
                };
                (source, true)
            }
            // a mapping that does not hold is not presented as the database's shape
            Lookup::Found(..) | Lookup::NotFound => (FragmentSource::Approximate(Why::NotInDatabase), false),
            Lookup::Pending => (FragmentSource::Approximate(Why::LookingUp), false),
            Lookup::Offline => (FragmentSource::Approximate(Why::Offline), false),
        };
        sources.push((frag.clone(), source));
        groups.push((frag, matched));
    }
    // R10 (a): a matched group the push cannot clear goes back to the embedding (honest, never an
    // overlap), and the rest are pushed again; each round unmatches at least one group, so this ends
    while groups.iter().any(|g| g.1) {
        push_apart(&mut x, &groups);
        let pts = |g: &[usize]| g.iter().map(|&i| x[i]).collect::<Vec<V3>>();
        let stuck: Vec<usize> = (0..groups.len())
            .filter(|&a| groups[a].1 && (0..groups.len()).any(|b| b != a && closest(&pts(&groups[a].0), &pts(&groups[b].0)) < FRAGMENT_GAP))
            .collect();
        if stuck.is_empty() {
            break;
        }
        for a in stuck {
            for &i in &groups[a].0 {
                x[i] = x0[i];
            }
            groups[a].1 = false;
            if let Some(s) = sources.iter_mut().find(|s| s.0 == groups[a].0) {
                s.1 = FragmentSource::Approximate(Why::NoRoom);
            }
        }
    }
    let matched = groups.iter().any(|g| g.1);
    (x, sources, matched)
}

/// `m` is a one-to-one map of `frag` onto `r`'s atoms that keeps every element and formal charge
/// and takes every drawn bond of the fragment to a record bond (orders may differ: the Kekulé
/// case, [`iso`]); a drawn bond leaving the fragment fails it. A source's answer is checked here
/// whatever it claims, so a wrong one is never presented as the database's shape.
fn valid_mapping(d: &Drawing, frag: &[usize], r: &Record, m: &[usize]) -> bool {
    let n = r.z.len();
    if m.len() != frag.len() || n != frag.len() || r.xyz.len() != n || r.charge.len() != n {
        return false;
    }
    let mut used = vec![false; n];
    let atoms_agree = frag.iter().zip(m).all(|(&a, &k)| k < n && !std::mem::replace(&mut used[k], true) && r.z[k] == d.atoms[a].z && r.charge[k] == d.atoms[a].charge);
    if !atoms_agree {
        return false;
    }
    let image = |a: usize| frag.iter().position(|&x| x == a).map(|i| m[i]);
    d.bonds.iter().all(|b| match (image(b.a), image(b.b)) {
        (Some(p), Some(q)) => r.bonds.iter().any(|&(u, v, _)| (u, v) == (p, q) || (u, v) == (q, p)),
        (None, None) => true,
        _ => false,
    })
}

/// The fragment's atoms at the record's shape (Bohr), its centroid where the embedding put the
/// fragment's.
fn put_record(x: &mut [V3], frag: &[usize], r: &Record, m: &[usize]) {
    let p: Vec<V3> = m.iter().map(|&k| r.xyz[k].map(units::angstrom_to_bohr)).collect();
    let (ce, cr) = (centroid(frag.iter().map(|&i| x[i])), centroid(p.iter().copied()));
    for (&i, q) in frag.iter().zip(&p) {
        for c in 0..3 {
            x[i][c] = q[c] - cr[c] + ce[c];
        }
    }
}

fn centroid(v: impl Iterator<Item = V3>) -> V3 {
    let (mut s, mut n) = ([0.0; 3], 0usize);
    for p in v {
        for c in 0..3 {
            s[c] += p[c];
        }
        n += 1;
    }
    s.map(|c| c / n.max(1) as f64)
}

/// Moves the matched groups (translations only) until every pair of atoms in different groups with
/// at least one matched is [`FRAGMENT_GAP`] apart: each short pair is pushed along the line between
/// the groups' centroids, split evenly when both may move.
fn push_apart(x: &mut [V3], groups: &[(Vec<usize>, bool)]) {
    let pts = |x: &[V3], g: &[usize]| g.iter().map(|&i| x[i]).collect::<Vec<V3>>();
    for _ in 0..MAX_PUSH_PASSES {
        let mut moved = false;
        for a in 0..groups.len() {
            for b in a + 1..groups.len() {
                let ((ga, ma), (gb, mb)) = (&groups[a], &groups[b]);
                if !(*ma || *mb) {
                    continue;
                }
                let (pa, pb) = (pts(x, ga), pts(x, gb));
                let gap = closest(&pa, &pb);
                if gap >= FRAGMENT_GAP {
                    continue;
                }
                let (ca, cb) = (centroid(pa.into_iter()), centroid(pb.into_iter()));
                let v = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let dir = if len > 1e-9 { v.map(|c| c / len) } else { [1.0, 0.0, 0.0] };
                let s = FRAGMENT_GAP - gap + PUSH_SLACK;
                let (sa, sb) = match (ma, mb) {
                    (true, true) => (-0.5 * s, 0.5 * s),
                    (true, false) => (-s, 0.0),
                    _ => (0.0, s),
                };
                for (g, t) in [(ga, sa), (gb, sb)] {
                    for &i in g {
                        for c in 0..3 {
                            x[i][c] += t * dir[c];
                        }
                    }
                }
                moved = true;
            }
        }
        if !moved {
            return;
        }
    }
}

/// One fragment's label (spec §4 Goal 5, verbatim).
pub fn fragment_label(s: &FragmentSource) -> String {
    match s {
        FragmentSource::PubChem { name, cid, stereo_note } => {
            let mut t = format!("shape: PubChem \"{name}\" (CID {cid})");
            if *stereo_note {
                t += &format!(" · stereo as in the database (CID {cid})");
            }
            t
        }
        FragmentSource::Lira => "shape: relaxed with LIRA (LSDA, Accurate)".into(),
        FragmentSource::Approximate(Why::NotInDatabase) => "shape: approximate (not in the database) — Relax after recommended".into(),
        FragmentSource::Approximate(Why::LookingUp) => "looking up online…".into(),
        FragmentSource::Approximate(Why::Offline) => "shape: approximate (offline)".into(),
        FragmentSource::Approximate(Why::NoRoom) => "shape: approximate (no room for the database shape) — Relax after recommended".into(),
    }
}

/// A fragment of `s` is still being looked up online: Insert and Replace wait (spec §7).
pub fn looking_up(s: &Shaped) -> bool {
    s.sources.iter().any(|(_, f)| *f == FragmentSource::Approximate(Why::LookingUp))
}

/// The footer's shape line (spec §8): one fragment's label, or for several the summary
/// "shape: k from PubChem, j relaxed with LIRA, m approximate" (zero parts left out; a fragment
/// still being looked up counts as approximate, and "looking up online…" follows). Empty when
/// nothing was looked up (no drawing, or lone atoms only).
pub fn footer_label(s: &Shaped) -> String {
    match s.sources.as_slice() {
        [] => String::new(),
        [(_, one)] => fragment_label(one),
        all => {
            let count = |f: &dyn Fn(&FragmentSource) -> bool| all.iter().filter(|(_, s)| f(s)).count();
            let k = count(&|s| matches!(s, FragmentSource::PubChem { .. }));
            let j = count(&|s| matches!(s, FragmentSource::Lira));
            let m = count(&|s| matches!(s, FragmentSource::Approximate(_)));
            let parts: Vec<String> = [(k, "from PubChem"), (j, "relaxed with LIRA"), (m, "approximate")].iter().filter(|p| p.0 > 0).map(|(n, w)| format!("{n} {w}")).collect();
            let mut t = format!("shape: {}", parts.join(", "));
            if all.iter().any(|(_, s)| *s == FragmentSource::Approximate(Why::LookingUp)) {
                t += " · looking up online…";
            }
            t
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn rec(name: &str) -> &'static Record {
        Library::shipped().records().iter().find(|r| r.name == name).unwrap_or_else(|| panic!("{name} not in the shipped list"))
    }

    /// Draws `r`'s graph into `d` at its canvas layout shifted by `at`, atoms in `perm` order
    /// (drawn atom k is record atom perm[k]); returns the drawn indices in record order.
    pub(crate) fn draw_record(d: &mut Drawing, r: &Record, at: [f32; 2], perm: &[usize]) -> Vec<usize> {
        let mut idx = vec![0; r.z.len()];
        for &k in perm {
            let i = d.add_atom(r.z[k], [r.xy[k][0] + at[0], r.xy[k][1] + at[1]]);
            d.atoms[i].charge = r.charge[k];
            idx[k] = i;
        }
        for &(a, b, o) in &r.bonds {
            let k = d.bond(idx[a], idx[b]).unwrap();
            d.bonds[k].order = o;
        }
        idx
    }

    fn identity(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn dist(p: V3, q: V3) -> f64 {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    /// Every pair of `frag`'s atoms is as far apart as the mapped record atoms (Å → Bohr).
    fn assert_record_shape(x: &[V3], d: &Drawing, frag: &[usize], r: &Record) {
        let iso::MapResult::Mapped(m) = iso::map_fragment(d, frag, r) else { panic!("{} does not map", r.name) };
        for i in 0..frag.len() {
            for j in i + 1..frag.len() {
                let want = units::angstrom_to_bohr(dist(r.xyz[m[i]], r.xyz[m[j]]));
                let got = dist(x[frag[i]], x[frag[j]]);
                assert!((got - want).abs() < 1e-9, "{} atoms {i}, {j}: {got} vs {want}", r.name);
            }
        }
    }

    /// A source answering from a closure.
    struct By<F: Fn(&Drawing, &[usize]) -> Lookup>(F);

    impl<F: Fn(&Drawing, &[usize]) -> Lookup> RecordSource for By<F> {
        fn find(&self, d: &Drawing, frag: &[usize]) -> Lookup {
            (self.0)(d, frag)
        }
    }

    /// Found when `r` maps onto the fragment, else `otherwise`.
    fn found_or(d: &Drawing, frag: &[usize], r: &Record, otherwise: Lookup) -> Lookup {
        match iso::map_fragment(d, frag, r) {
            iso::MapResult::Mapped(m) => Lookup::Found(r.clone(), m),
            _ => otherwise,
        }
    }

    #[test]
    fn a_matched_hand_off_uses_the_record_shape() {
        let eth = rec("ethanol");
        // drawn in a shuffled order, at its canvas layout
        let mut d = Drawing::default();
        let n = eth.z.len();
        draw_record(&mut d, eth, [0.0, 0.0], &[3, 0, 8, 7, 1, 5, 2, 6, 4]);
        let s = shape(&d, &LocalOnly(Library::shipped()));
        assert_eq!(s.positions.len(), n);
        assert_record_shape(&s.positions, &d, &identity(n), eth);
        assert_eq!(s.sources, vec![(identity(n), FragmentSource::PubChem { name: "ethanol".into(), cid: 702, stereo_note: false })]);
        // centred like the embedding
        let c = centroid(s.positions.iter().copied());
        assert!(c.iter().all(|v| v.abs() < 1e-12), "{c:?}");
    }

    #[test]
    fn an_unmatched_drawing_is_bit_identical_to_today() {
        // H₃⁺ (a triangle, one H +1): one fragment
        let mut h3 = Drawing::default();
        for p in [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]] {
            h3.add_atom(1, p);
        }
        h3.bond(0, 1);
        h3.bond(1, 2);
        h3.bond(2, 0);
        h3.atoms[0].charge = 1;
        // two fragments: H₃⁺ and a He≡He that no database holds
        let mut two = h3.clone();
        let a = two.add_atom(2, [3.0, 0.0]);
        let b = two.add_atom(2, [4.0, 0.0]);
        let k = two.bond(a, b).unwrap();
        two.bonds[k].order = 3;
        // H₃⁺ joins the database with the LIRA entries (Task 7), so the property is pinned with
        // sources that never find, in every not-found way, and the shipped list on the He₂
        let never = By(|_: &Drawing, _: &[usize]| Lookup::NotFound);
        let pending = By(|_: &Drawing, _: &[usize]| Lookup::Pending);
        let offline = By(|_: &Drawing, _: &[usize]| Lookup::Offline);
        for d in [&h3, &two] {
            let today = embed::embed(d);
            for src in [&never as &dyn RecordSource, &pending, &offline] {
                let s = shape(d, src);
                let bits = |v: &[V3]| v.iter().flat_map(|p| p.map(f64::to_bits)).collect::<Vec<_>>();
                assert_eq!(bits(&s.positions), bits(&today), "bit for bit");
            }
        }
        let mut he2 = Drawing::default();
        let a = he2.add_atom(2, [0.0, 0.0]);
        let b = he2.add_atom(2, [1.0, 0.0]);
        let k = he2.bond(a, b).unwrap();
        he2.bonds[k].order = 3;
        let lone = he2.add_atom(10, [3.0, 0.0]);
        let s = shape(&he2, &LocalOnly(Library::shipped()));
        assert_eq!(s.positions.iter().map(|p| p.map(f64::to_bits)).collect::<Vec<_>>(), embed::embed(&he2).iter().map(|p| p.map(f64::to_bits)).collect::<Vec<_>>());
        assert_eq!(s.sources, vec![(vec![a, b], FragmentSource::Approximate(Why::NotInDatabase))], "the lone Ne is not labelled ({lone})");
    }

    #[test]
    fn two_waters_and_a_lone_atom_keep_their_gaps() {
        let w = rec("water");
        // two waters drawn almost on top of each other, and a lone Na between them
        let mut d = Drawing::default();
        let w1 = draw_record(&mut d, w, [0.0, 0.0], &[0, 1, 2]);
        let na = d.add_atom(11, [0.6, 0.2]);
        let w2 = draw_record(&mut d, w, [0.3, 0.4], &[2, 0, 1]);
        let src = LocalOnly(Library::shipped());
        let s = shape(&d, &src);
        let frags = iso::fragments(&d);
        assert_eq!(frags.len(), 3);
        // the push is exercised: the records at the embedding's centroids would be too close
        let mut unpushed = embed::embed(&d);
        for f in [&w1, &w2] {
            let mut f = f.clone();
            f.sort_unstable();
            let iso::MapResult::Mapped(m) = iso::map_fragment(&d, &f, w) else { panic!() };
            put_record(&mut unpushed, &f, w, &m);
        }
        let pts = |f: &[usize]| f.iter().map(|&i| unpushed[i]).collect::<Vec<V3>>();
        assert!(closest(&pts(&w1), &pts(&w2)) < FRAGMENT_GAP, "{}", closest(&pts(&w1), &pts(&w2)));
        // each water is the record's
        for f in [&w1, &w2] {
            let mut f = f.clone();
            f.sort_unstable();
            assert_record_shape(&s.positions, &d, &f, w);
        }
        assert_eq!(s.sources.len(), 2, "the lone Na has no label");
        assert!(s.sources.iter().all(|(_, f)| matches!(f, FragmentSource::PubChem { cid: 962, .. })));
        // every pair of atoms in different fragments ≥ 3 Bohr
        for (a, fa) in frags.iter().enumerate() {
            for fb in &frags[a + 1..] {
                for &i in fa {
                    for &j in fb {
                        assert!(dist(s.positions[i], s.positions[j]) >= FRAGMENT_GAP, "atoms {i}, {j}: {}", dist(s.positions[i], s.positions[j]));
                    }
                }
            }
        }
        // the lone atom stays where the whole drawing's embedding put it, shifted only by the final
        // centring
        let (raw, _, matched) = place(&d, &src);
        assert!(matched);
        assert_eq!(raw[na].map(f64::to_bits), embed::embed(&d)[na].map(f64::to_bits));
        let mut centred = raw.clone();
        embed::centre(&mut centred);
        assert_eq!(centred, s.positions);
        let shift = centroid(raw.iter().copied());
        for c in 0..3 {
            assert!((s.positions[na][c] - (raw[na][c] - shift[c])).abs() < 1e-12);
        }
    }

    #[test]
    fn labels_read_as_the_spec_says() {
        let (eth, water) = (rec("ethanol"), rec("water"));
        let mut d = Drawing::default();
        draw_record(&mut d, eth, [0.0, 0.0], &identity(eth.z.len()));
        let local = LocalOnly(Library::shipped());
        assert_eq!(footer_label(&shape(&d, &local)), "shape: PubChem \"ethanol\" (CID 702)");
        // our own relaxed entry
        let mut lira = water.clone();
        lira.source = Source::Lira { functional: "LSDA".into(), quality: "Accurate".into(), ionic: "accepted".into(), multiplicity: 1, max_force: 3e-4, commit: "test".into() };
        let mut wd = Drawing::default();
        draw_record(&mut wd, water, [0.0, 0.0], &identity(3));
        let lira_src = By(|d: &Drawing, f: &[usize]| found_or(d, f, &lira, Lookup::NotFound));
        assert_eq!(footer_label(&shape(&wd, &lira_src)), "shape: relaxed with LIRA (LSDA, Accurate)");
        // not in the database
        let mut odd = Drawing::default();
        odd.add_atom(6, [0.0, 0.0]);
        odd.add_atom(6, [1.0, 0.0]);
        odd.bond(0, 1);
        assert_eq!(footer_label(&shape(&odd, &local)), "shape: approximate (not in the database) — Relax after recommended");
        // in flight, and offline
        assert_eq!(footer_label(&shape(&odd, &By(|_: &Drawing, _: &[usize]| Lookup::Pending))), "looking up online…");
        assert_eq!(footer_label(&shape(&odd, &By(|_: &Drawing, _: &[usize]| Lookup::Offline))), "shape: approximate (offline)");
        // several fragments: ethanol (PubChem), water (LIRA), C₂ (approximate)
        let mut mixed = d.clone();
        draw_record(&mut mixed, water, [5.0, 0.0], &identity(3));
        let c = mixed.add_atom(6, [9.0, 0.0]);
        let c2 = mixed.add_atom(6, [10.0, 0.0]);
        mixed.bond(c, c2);
        let mixed_src = By(|d: &Drawing, f: &[usize]| found_or(d, f, eth, found_or(d, f, &lira, Lookup::NotFound)));
        let s = shape(&mixed, &mixed_src);
        assert_eq!(footer_label(&s), "shape: 1 from PubChem, 1 relaxed with LIRA, 1 approximate");
        assert_eq!(
            s.sources.iter().map(|(_, f)| fragment_label(f)).collect::<Vec<_>>(),
            ["shape: PubChem \"ethanol\" (CID 702)", "shape: relaxed with LIRA (LSDA, Accurate)", "shape: approximate (not in the database) — Relax after recommended"]
        );
        // zero parts are left out; one in flight counts as approximate and says so
        let mut ww = wd.clone();
        draw_record(&mut ww, water, [4.0, 0.0], &identity(3));
        assert_eq!(footer_label(&shape(&ww, &local)), "shape: 2 from PubChem");
        let first_pending = By(|d: &Drawing, f: &[usize]| if f[0] == 0 { Lookup::Pending } else { found_or(d, f, water, Lookup::NotFound) });
        assert_eq!(footer_label(&shape(&ww, &first_pending)), "shape: 1 from PubChem, 1 approximate · looking up online…");
        // a stereo-specific record: the drawing did not decide its stereo
        let mut stereo = eth.clone();
        stereo.stereo_defined = true;
        let stereo_src = By(|d: &Drawing, f: &[usize]| found_or(d, f, &stereo, Lookup::NotFound));
        assert_eq!(footer_label(&shape(&d, &stereo_src)), "shape: PubChem \"ethanol\" (CID 702) · stereo as in the database (CID 702)");
        // nothing to label
        assert_eq!(footer_label(&shape(&Drawing::default(), &local)), "");
    }

    #[test]
    fn a_mapping_that_does_not_hold_is_approximate() {
        // a source claiming water's record for a drawn O–H–H chain with a bad mapping (O → H)
        let water = rec("water");
        let mut d = Drawing::default();
        draw_record(&mut d, water, [0.0, 0.0], &identity(3));
        let liar = By(|_: &Drawing, _: &[usize]| Lookup::Found(water.clone(), vec![1, 0, 2]));
        let s = shape(&d, &liar);
        assert_eq!(s.sources[0].1, FragmentSource::Approximate(Why::NotInDatabase));
        assert_eq!(s.positions, embed::embed(&d));
        // nor one of the wrong length or not one-to-one
        for m in [vec![0, 1], vec![0, 1, 1]] {
            let bad = By(move |_: &Drawing, _: &[usize]| Lookup::Found(water.clone(), m.clone()));
            assert_eq!(shape(&d, &bad).sources[0].1, FragmentSource::Approximate(Why::NotInDatabase));
        }
    }

    /// R10 (b): the elements alone do not make a mapping. A same-element record whose O carries a
    /// charge the drawing's does not, or whose bonds join other atoms than the drawn ones, is
    /// refused, whatever its source claims.
    #[test]
    fn a_wrong_charge_or_bond_record_is_refused() {
        let water = rec("water");
        let mut d = Drawing::default();
        draw_record(&mut d, water, [0.0, 0.0], &identity(3));
        let mut charged = water.clone();
        charged.charge[0] = -1;
        let wrong_charge = By(|_: &Drawing, _: &[usize]| Lookup::Found(charged.clone(), identity(3)));
        let s = shape(&d, &wrong_charge);
        assert_eq!(s.sources[0].1, FragmentSource::Approximate(Why::NotInDatabase));
        assert_eq!(s.positions, embed::embed(&d));
        // H–O–H drawn, the "record" bonded O–H and H–H: same elements, same bond count
        let mut rewired = water.clone();
        rewired.bonds = vec![(0, 1, 1), (1, 2, 1)];
        let wrong_bonds = By(|_: &Drawing, _: &[usize]| Lookup::Found(rewired.clone(), identity(3)));
        assert_eq!(shape(&d, &wrong_bonds).sources[0].1, FragmentSource::Approximate(Why::NotInDatabase));
        // the record as it is still maps
        let right = By(|_: &Drawing, _: &[usize]| Lookup::Found(water.clone(), identity(3)));
        assert!(matches!(shape(&d, &right).sources[0].1, FragmentSource::PubChem { .. }));
    }

    /// R10 (a): a matched fragment that the push cannot clear of two lone atoms on either side of it
    /// (exactly in line, so every push is along that line and the pushes undo each other) goes back
    /// to the embedding's positions, labelled approximate — never handed off overlapping them.
    #[test]
    fn a_fragment_trapped_between_two_lone_atoms_falls_back() {
        // Na · H–H · Na, the embedding given by hand: the Nas 5 Bohr apart, the H₂ between them
        let mut d = Drawing::default();
        let a = d.add_atom(11, [0.0, 0.0]);
        let h1 = d.add_atom(1, [1.0, 0.0]);
        let h2 = d.add_atom(1, [2.0, 0.0]);
        let b = d.add_atom(11, [3.0, 0.0]);
        d.bond(h1, h2);
        let x0: Vec<V3> = vec![[-2.5, 0.0, 0.0], [-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [2.5, 0.0, 0.0]];
        // a record of H₂ (0.74 Å) that every lookup returns
        let h2rec = Record {
            name: "hydrogen".into(),
            synonyms: vec![],
            source: Source::PubChem { cid: 783 },
            z: vec![1, 1],
            charge: vec![0, 0],
            bonds: vec![(0, 1, 1)],
            xyz: vec![[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]],
            xy: vec![[-0.5, 0.0], [0.5, 0.0]],
            stereo_defined: false,
        };
        let src = By(|d: &Drawing, f: &[usize]| found_or(d, f, &h2rec, Lookup::NotFound));
        let (x, sources, matched) = place_from(&d, &src, x0.clone());
        assert_eq!(sources, vec![(vec![h1, h2], FragmentSource::Approximate(Why::NoRoom))]);
        assert!(!matched, "nothing is left matched: no centring");
        assert_eq!(x.iter().map(|p| p.map(f64::to_bits)).collect::<Vec<_>>(), x0.iter().map(|p| p.map(f64::to_bits)).collect::<Vec<_>>(), "the embedding, bit for bit");
        assert_eq!(fragment_label(&sources[0].1), "shape: approximate (no room for the database shape) — Relax after recommended");
        // with room (the Nas 12 Bohr apart) the same fragment takes the record's shape
        let roomy: Vec<V3> = vec![[-6.0, 0.0, 0.0], [-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let (x, sources, matched) = place_from(&d, &src, roomy);
        assert!(matched && matches!(sources[0].1, FragmentSource::PubChem { cid: 783, .. }));
        assert!((dist(x[h1], x[h2]) - units::angstrom_to_bohr(0.74)).abs() < 1e-9);
        for lone in [a, b] {
            assert!(dist(x[lone], x[h1]).min(dist(x[lone], x[h2])) >= FRAGMENT_GAP);
        }
    }
}
