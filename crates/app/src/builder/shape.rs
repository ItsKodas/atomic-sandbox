//! The hand-off geometry and its labels (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md`
//! §6.1, §4 Goal 5 and §8): where Insert / Replace put the drawn atoms, and the footer's line saying
//! where that shape comes from.
//!
//! **The layout (§6.1).** [`embed::embed`] lays out the whole drawing once, exactly as before the
//! database existed. Each fragment of two or more atoms is then looked up through a
//! [`RecordSource`]; a found fragment's atoms take its record's 3D shape (Å → Bohr) through the
//! atom mapping, turned by the proper rotation (never a reflection) that best fits it to the
//! embedding's positions for that fragment ([`fit_rotation`]), then centred on the centroid the
//! embedding gave that fragment — so each fragment keeps the orientation it was drawn in. Only the matched
//! fragments then move, pushed apart along the line between fragment centroids until every pair of
//! atoms in different fragments is at least [`FRAGMENT_GAP`] apart, and the whole is centred last as
//! the embedding is. A matched fragment the push cannot clear within [`MAX_PUSH_PASSES`] (trapped
//! between atoms that do not move) goes back to the embedding's positions, labelled approximate
//! ([`Why::NoRoom`]), and the others are pushed again. Lone atoms and fragments not found keep the
//! embedding's positions; a drawing with no matched fragment is handed off bit for bit as the
//! embedding gives it.
//!
//! **Answers are checked** ([`valid_mapping`]): a source's mapping must keep every element and
//! formal charge and take the drawn bonds onto exactly the record's bonds, or the fragment is approximate.
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
/// and takes every drawn bond of the fragment to a record bond, the record having no other
/// (orders may differ: the Kekulé case, [`iso`]); a drawn bond leaving the fragment fails it. A source's answer is checked here
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
    let mut drawn = 0;
    let bonds_map = d.bonds.iter().all(|b| match (image(b.a), image(b.b)) {
        (Some(p), Some(q)) => {
            drawn += 1;
            r.bonds.iter().any(|&(u, v, _)| (u, v) == (p, q) || (u, v) == (q, p))
        }
        (None, None) => true,
        _ => false,
    });
    // and the record has no bond the drawing lacks (the map is one-to-one, so equal counts suffice)
    bonds_map && drawn == r.bonds.len()
}

/// The fragment's atoms at the record's shape (Bohr), turned by [`fit_rotation`] to best fit the
/// embedding's positions for them, its centroid where the embedding put the fragment's.
fn put_record(x: &mut [V3], frag: &[usize], r: &Record, m: &[usize]) {
    let bohr = |k: usize| r.xyz[k].map(units::angstrom_to_bohr);
    let p: Vec<V3> = m.iter().map(|&k| bohr(k)).collect();
    let (ce, cr) = (centroid(frag.iter().map(|&i| x[i])), centroid(p.iter().copied()));
    let p: Vec<V3> = p.iter().map(|v| sub(*v, cr)).collect();
    let q: Vec<V3> = frag.iter().map(|&i| sub(x[i], ce)).collect();
    // the record's own axes, from its atoms in record order (independent of the drawing's order)
    let all: Vec<V3> = (0..r.xyz.len()).map(bohr).collect();
    let ca = centroid(all.iter().copied());
    let axes = principal_axes(&all.iter().map(|v| sub(*v, ca)).collect::<Vec<_>>());
    let rot = fit_rotation(&p, &q, &axes);
    for (&i, v) in frag.iter().zip(&p) {
        let t = apply(&rot, *v);
        for c in 0..3 {
            x[i][c] = t[c] + ce[c];
        }
    }
}

type M3 = [[f64; 3]; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn apply(r: &M3, v: V3) -> V3 {
    [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
}

/// The weight of the tie-break term in [`fit_rotation`], relative to the record's Σ|p|² (Bohr²).
const TIE_BREAK: f64 = 1e-9;

/// The proper rotation `R` (det = +1, never a reflection: a reflection would flip a stereo record's
/// chirality) that best fits `R p_i` onto `q_i` in the least-squares sense — Horn's quaternion form
/// of the Kabsch fit: the unit quaternion is the top eigenvector of the symmetric 4×4 matrix built
/// from S = Σ p_i q_iᵀ. `p` (the record, centred) and `q` (the embedding, centred) are paired by
/// index.
///
/// **The tie-break (degenerate fits).** Where the drawing leaves part of the turn free, the free
/// part is decided by a term of weight [`TIE_BREAK`] × Σ|p|² that turns the record's own principal
/// axes `axes` (largest spread first, [`principal_axes`]) as near as allowed onto the scene's x, y
/// and z — so a free turn lays the record flat in the drawing's plane. The term is fixed per record,
/// so the answer is deterministic, and continuous wherever the fit is unique; its pull on a
/// determined fit is of order 1e-9 rad.
/// - a 2-atom or linear record: only the line's direction is fitted (onto the embedding's line,
///   its sense from the atom mapping); the turn about it is the tie-break's, and moves no atom;
/// - a flat (or near-flat) embedding, as a 2D drawing gives: S has rank 2, which still fixes a
///   unique proper rotation (the third axis follows from the first two), so a tiny out-of-plane
///   change in the embedding moves the answer by as little, without a flip between mirror frames;
/// - a straight-line embedding of a bent record: the record's best-matching direction goes onto the
///   line and the tie-break picks the turn about it.
fn fit_rotation(p: &[V3], q: &[V3], axes: &[V3; 3]) -> M3 {
    let mut s = [[0.0; 3]; 3];
    for (a, b) in p.iter().zip(q) {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += a[i] * b[j];
            }
        }
    }
    let eps = TIE_BREAK * p.iter().map(|v| dot(*v, *v)).sum::<f64>();
    for (k, w) in [3.0, 2.0, 1.0].into_iter().enumerate() {
        // as if the record's axis k (weight w) were a point paired with the scene's axis k
        for i in 0..3 {
            s[i][k] += eps * w * axes[k][i];
        }
    }
    let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
    let n = [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ];
    let (val, vec) = jacobi(n);
    let mut top = 0;
    for k in 1..4 {
        if val[k] > val[top] {
            top = k;
        }
    }
    let qn = (0..4).map(|i| vec[i][top] * vec[i][top]).sum::<f64>().sqrt();
    let [w, x, y, z] = [0, 1, 2, 3].map(|i| vec[i][top] / qn);
    [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (y * x + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w * w - x * x - y * y + z * z],
    ]
}

/// The principal axes of the centred points `p` (eigenvectors of Σ p pᵀ), largest spread first, as
/// a right-handed frame: each of the first two signed so the points' third moment along it is
/// positive (or, when that vanishes, so the first point off its normal plane lies on its + side),
/// the third their cross product. Deterministic for a given record.
pub(crate) fn principal_axes(p: &[V3]) -> [V3; 3] {
    let mut c = [[0.0; 3]; 3];
    for v in p {
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] += v[i] * v[j];
            }
        }
    }
    let (val, vec) = jacobi(c);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| val[b].total_cmp(&val[a]));
    let scale = p.iter().map(|v| dot(*v, *v)).sum::<f64>().sqrt().max(1e-300);
    let mut ax = [[0.0; 3]; 2];
    for (slot, &k) in order[..2].iter().enumerate() {
        let a = [vec[0][k], vec[1][k], vec[2][k]];
        let third: f64 = p.iter().map(|v| dot(*v, a).powi(3)).sum();
        let sign = if third.abs() > 1e-9 * scale.powi(3) {
            third.signum()
        } else {
            p.iter().map(|v| dot(*v, a)).find(|t| t.abs() > 1e-9 * scale).map_or(1.0, f64::signum)
        };
        ax[slot] = a.map(|c| c * sign);
    }
    [ax[0], ax[1], cross(ax[0], ax[1])]
}

/// The eigenvalues and eigenvectors (columns) of a small symmetric matrix, by cyclic Jacobi
/// rotations.
fn jacobi<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..N).flat_map(|i| (i + 1..N).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        let diag: f64 = (0..N).map(|i| a[i][i] * a[i][i]).sum();
        if off == 0.0 || off <= 1e-32 * diag {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (kp, kq) = (a[k][p], a[k][q]);
                    a[k][p] = c * kp - s * kq;
                    a[k][q] = s * kp + c * kq;
                }
                for k in 0..N {
                    let (pk, qk) = (a[p][k], a[q][k]);
                    a[p][k] = c * pk - s * qk;
                    a[q][k] = s * pk + c * qk;
                }
                for row in v.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
            }
        }
    }
    (std::array::from_fn(|i| a[i][i]), v)
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
        // every drawn bond is there, but the "record" has one more (H–H): a ring the drawing lacks
        let mut extra = water.clone();
        extra.bonds.push((1, 2, 1));
        let extra_bond = By(|_: &Drawing, _: &[usize]| Lookup::Found(extra.clone(), identity(3)));
        assert_eq!(shape(&d, &extra_bond).sources[0].1, FragmentSource::Approximate(Why::NotInDatabase));
        // the record as it is still maps
        let right = By(|_: &Drawing, _: &[usize]| Lookup::Found(water.clone(), identity(3)));
        assert!(matches!(shape(&d, &right).sources[0].1, FragmentSource::PubChem { .. }));
    }

    /// The angle (degrees) at `c` between `a` and `b`.
    fn angle_at(a: V3, c: V3, b: V3) -> f64 {
        let (u, v) = ([a[0] - c[0], a[1] - c[1], a[2] - c[2]], [b[0] - c[0], b[1] - c[1], b[2] - c[2]]);
        let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        (dot / (dist(a, c) * dist(b, c))).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// `r` drawn at its canvas layout turned by `deg` (canvas coordinates) and shifted by `at`;
    /// returns the drawn indices in record order.
    fn draw_turned(d: &mut Drawing, r: &Record, at: [f32; 2], deg: f32, mirror: bool) -> Vec<usize> {
        let (s, c) = deg.to_radians().sin_cos();
        let mut t = r.clone();
        for p in &mut t.xy {
            let x = if mirror { -p[0] } else { p[0] };
            *p = [c * x - s * p[1], s * x + c * p[1]];
        }
        draw_record(d, &t, at, &identity(t.z.len()))
    }

    /// The user's hand-test (2026-11-25): water drawn above HF, HF's H pointing at water's O and no
    /// bond between them (the H₂O···HF hydrogen bond). Each fragment takes its record's shape, turned
    /// to the drawing, so HF's H still points at the O — not left in PubChem's frame, where F came
    /// out nearest the O.
    #[test]
    fn a_drawn_hydrogen_bond_keeps_its_orientation() {
        let (w, hf) = (rec("water"), rec("hydrogen fluoride"));
        let mut d = Drawing::default();
        // water on top, its Hs pointing up (canvas y points down)
        let o = d.add_atom(8, [0.0, 0.0]);
        let h1 = d.add_atom(1, [-0.8, -0.6]);
        let h2 = d.add_atom(1, [0.8, -0.6]);
        d.bond(o, h1);
        d.bond(o, h2);
        // HF below, H up toward the O
        let h = d.add_atom(1, [0.0, 1.6]);
        let f = d.add_atom(9, [0.0, 2.6]);
        d.bond(h, f);
        let s = shape(&d, &LocalOnly(Library::shipped()));
        assert!(matches!(s.sources[0].1, FragmentSource::PubChem { cid: 962, .. }), "{:?}", s.sources);
        assert!(matches!(s.sources[1].1, FragmentSource::PubChem { cid: 14917, .. }), "{:?}", s.sources);
        // the records' shapes: a rotation keeps every distance
        assert_record_shape(&s.positions, &d, &[o, h1, h2], w);
        assert_record_shape(&s.positions, &d, &[h, f], hf);
        let x = &s.positions;
        assert!(dist(x[h], x[o]) < dist(x[f], x[o]), "H···O {} vs F···O {}", dist(x[h], x[o]), dist(x[f], x[o]));
        let a = angle_at(x[o], x[h], x[f]);
        assert!(a > 150.0, "O···H–F {a}°");
        // and the water's Hs still point away from the HF
        for hw in [h1, h2] {
            assert!(dist(x[hw], x[f]) > dist(x[o], x[f]), "water H {hw}");
        }
    }

    /// A lone matched fragment turned on the canvas comes out turned the same way: the record's C–O
    /// bond, seen from +z, points where the drawn one does (canvas y down, the scene's y up). Formic
    /// acid is planar; it is drawn as its record's own shape seen from +z (the shipped canvas layout
    /// draws the other O–H conformer, which the fit can only meet halfway).
    #[test]
    fn a_single_fragment_follows_the_drawing() {
        let mut fa = rec("formic acid").clone();
        fa.xy = fa.xyz.iter().map(|p| [(p[0] / 1.3) as f32, (-p[1] / 1.3) as f32]).collect();
        let fa = &fa;
        let c = fa.z.iter().position(|&z| z == 6).unwrap();
        for o in (0..fa.z.len()).filter(|&k| fa.z[k] == 8) {
            for deg in [0.0f32, 37.0, 90.0, 180.0, 270.0] {
                let mut d = Drawing::default();
                let idx = draw_turned(&mut d, fa, [0.0, 0.0], deg, false);
                let s = shape(&d, &LocalOnly(Library::shipped()));
                assert!(matches!(s.sources[0].1, FragmentSource::PubChem { .. }));
                assert_record_shape(&s.positions, &d, &identity(fa.z.len()), fa);
                let (pc, po) = (d.atoms[idx[c]].pos, d.atoms[idx[o]].pos);
                let drawn = [(po[0] - pc[0]) as f64, -(po[1] - pc[1]) as f64];
                let (xc, xo) = (s.positions[idx[c]], s.positions[idx[o]]);
                let got = [xo[0] - xc[0], xo[1] - xc[1]];
                let full = dist(xc, xo);
                let in_plane = got[0].hypot(got[1]);
                assert!(in_plane > 0.9 * full, "{deg}°: the planar molecule lies in the drawing's plane ({in_plane} of {full})");
                let turn = (got[1].atan2(got[0]) - drawn[1].atan2(drawn[0])).to_degrees();
                let turn = (turn + 540.0).rem_euclid(360.0) - 180.0;
                assert!(turn.abs() < 15.0, "{deg}°, O {o}: C–O off the drawing by {turn}°");
            }
        }
    }

    /// The turn is a proper rotation, never a reflection: a stereo-defined record (L-alanine), drawn
    /// as laid out and mirrored on the canvas, keeps its chirality — the signed volume of the
    /// stereocentre's neighbours is the record's.
    #[test]
    fn the_fit_is_a_rotation_not_a_reflection() {
        let ala = rec("L-alanine");
        assert!(ala.stereo_defined);
        let n = ala.z.iter().position(|&z| z == 7).unwrap();
        let neighbours = |k: usize| -> Vec<usize> {
            let mut v: Vec<usize> = ala.bonds.iter().filter_map(|&(a, b, _)| if a == k { Some(b) } else if b == k { Some(a) } else { None }).collect();
            v.sort_unstable();
            v
        };
        let ca = neighbours(n).into_iter().find(|&k| ala.z[k] == 6).unwrap();
        let nb = neighbours(ca);
        assert_eq!(nb.len(), 4);
        let vol = |p: &[V3]| {
            let (a, b, c, e) = (p[0], p[1], p[2], p[3]);
            let (u, v, w) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]], [e[0] - a[0], e[1] - a[1], e[2] - a[2]]);
            u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])
        };
        let want = vol(&nb.iter().map(|&k| ala.xyz[k].map(units::angstrom_to_bohr)).collect::<Vec<_>>());
        assert!(want.abs() > 0.5, "{want}");
        for mirror in [false, true] {
            for deg in [0.0f32, 90.0, 200.0] {
                let mut d = Drawing::default();
                let idx = draw_turned(&mut d, ala, [0.0, 0.0], deg, mirror);
                let s = shape(&d, &LocalOnly(Library::shipped()));
                assert!(matches!(s.sources[0].1, FragmentSource::PubChem { .. }), "{:?}", s.sources);
                assert_record_shape(&s.positions, &d, &identity(ala.z.len()), ala);
                let got = vol(&nb.iter().map(|&k| s.positions[idx[k]]).collect::<Vec<_>>());
                assert!((got - want).abs() < 1e-6, "mirror {mirror}, {deg}°: {got} vs {want}");
            }
        }
    }

    fn det(r: &M3) -> f64 {
        dot(r[0], cross(r[1], r[2]))
    }

    fn turn(axis: V3, deg: f64) -> M3 {
        let n = (dot(axis, axis)).sqrt();
        let [x, y, z] = axis.map(|c| c / n);
        let (s, c) = deg.to_radians().sin_cos();
        let t = 1.0 - c;
        [[c + x * x * t, x * y * t - z * s, x * z * t + y * s], [y * x * t + z * s, c + y * y * t, y * z * t - x * s], [z * x * t - y * s, z * y * t + x * s, c + z * z * t]]
    }

    fn centred(v: &[V3]) -> Vec<V3> {
        let c = centroid(v.iter().copied());
        v.iter().map(|p| sub(*p, c)).collect()
    }

    fn close(a: &M3, b: &M3, tol: f64) -> bool {
        (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() < tol))
    }

    /// A chiral, non-planar cloud with distinct spreads.
    fn cloud() -> Vec<V3> {
        centred(&[[0.0, 0.0, 0.0], [2.1, 0.3, -0.2], [-0.4, 1.7, 0.5], [0.6, -0.9, 1.4], [-1.8, -0.6, -0.9]])
    }

    /// The fit recovers a turn exactly; a mirror image still gets a proper rotation; and the result
    /// is orthonormal.
    #[test]
    fn the_fit_recovers_a_turn_and_never_reflects() {
        let p = cloud();
        let axes = principal_axes(&p);
        assert!((det(&[axes[0], axes[1], axes[2]]) - 1.0).abs() < 1e-12, "a right-handed frame");
        for (axis, deg) in [([1.0, 2.0, 3.0], 73.0), ([0.0, 0.0, 1.0], 180.0), ([-1.0, 0.5, 0.2], 179.9), ([0.3, 0.1, -1.0], 0.0)] {
            let r0 = turn(axis, deg);
            let q: Vec<V3> = p.iter().map(|v| apply(&r0, *v)).collect();
            let r = fit_rotation(&p, &q, &axes);
            assert!(close(&r, &r0, 1e-7), "{axis:?} {deg}°: {r:?} vs {r0:?}");
        }
        // the mirror image (x → −x): the best proper rotation, not the reflection
        let q: Vec<V3> = p.iter().map(|v| [-v[0], v[1], v[2]]).collect();
        let r = fit_rotation(&p, &q, &axes);
        assert!((det(&r) - 1.0).abs() < 1e-12, "{}", det(&r));
        for i in 0..3 {
            for j in 0..3 {
                let rrt = dot(r[i], r[j]);
                assert!((rrt - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    /// The degenerate fits (module doc of [`fit_rotation`]): a 2-atom record goes onto the drawn
    /// line, its sense kept; a flat embedding gives a proper rotation that does not flip when the
    /// embedding's tiny out-of-plane offsets change sign; a straight-line embedding of a bent record
    /// is decided the same way every time.
    #[test]
    fn degenerate_fits_are_proper_and_stable() {
        // two atoms: record along z, drawn along (1, 1, 0)
        let p = vec![[0.0, 0.0, -0.9], [0.0, 0.0, 0.9]];
        let q = vec![[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]];
        let axes = principal_axes(&p);
        let r = fit_rotation(&p, &q, &axes);
        assert!((det(&r) - 1.0).abs() < 1e-12);
        let s = 0.9 / 2f64.sqrt();
        for (a, want) in p.iter().zip([[-s, -s, 0.0], [s, s, 0.0]]) {
            let got = apply(&r, *a);
            assert!((0..3).all(|c| (got[c] - want[c]).abs() < 1e-6), "{got:?} vs {want:?}");
        }
        // reversed: the sense follows the mapping
        let r = fit_rotation(&p, &[q[1], q[0]], &axes);
        assert!(apply(&r, p[0])[0] > 0.6);
        // a flat embedding of the chiral cloud, offsets ±1e-6 out of plane
        let p = cloud();
        let axes = principal_axes(&p);
        let flat = |sign: f64| p.iter().enumerate().map(|(i, v)| [v[0], v[1], sign * 1e-6 * (i as f64 - 2.0)]).collect::<Vec<V3>>();
        let (up, down) = (fit_rotation(&p, &flat(1.0), &axes), fit_rotation(&p, &flat(-1.0), &axes));
        assert!((det(&up) - 1.0).abs() < 1e-12 && (det(&down) - 1.0).abs() < 1e-12);
        assert!(close(&up, &down, 1e-4), "{up:?} vs {down:?}");
        // and exactly flat, its mirror drawn in the plane: still proper
        let mirrored: Vec<V3> = p.iter().map(|v| [-v[0], v[1], 0.0]).collect();
        assert!((det(&fit_rotation(&p, &mirrored, &axes)) - 1.0).abs() < 1e-12);
        // a bent record on a straight line: deterministic, proper, its best direction on the line
        let line: Vec<V3> = centred(&(0..5).map(|i| [i as f64, 0.0, 0.0]).collect::<Vec<_>>());
        let (a, b) = (fit_rotation(&p, &line, &axes), fit_rotation(&p, &line, &axes));
        assert_eq!(a, b);
        assert!((det(&a) - 1.0).abs() < 1e-12);
        let w: V3 = p.iter().zip(&line).fold([0.0; 3], |acc, (v, l)| [acc[0] + l[0] * v[0], acc[1] + l[0] * v[1], acc[2] + l[0] * v[2]]);
        let along = apply(&a, w);
        assert!(along[0] > 0.999 * dot(w, w).sqrt(), "{along:?}");
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
