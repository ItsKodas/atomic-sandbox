//! The LIRA list (spec `docs/superpowers/specs/2026-11-25-builder-database-design.md` §5.1): the
//! molecules within H–Ar that PubChem has no 3D shape for, as graphs (`crates/app/data/lira-molecules.txt`,
//! its header documents the format). The dev-only generator (`worker::tests::generate_lira_shapes`) relaxes
//! each with the app's own Relax and writes it into the shipped data file as a `source lira` row; the
//! records tests check those rows against this list (`records::tests::lira_rows_are_relaxed_and_traceable`).
//! Test builds only: nothing here ships, the rows it produced do.
//!
//! **Rule 1 (spec §3).** Like every record, a LIRA row only places nuclei in the builder.
use super::records::{canvas_layout, Record, Source};
use super::{chem, shape, Atom, Bond, Drawing};
use std::collections::VecDeque;
use units::elements::z_of_symbol;

type V3 = [f64; 3];

/// The list (`crates/app/data/lira-molecules.txt`).
pub const LIST: &str = include_str!("../../data/lira-molecules.txt");

/// One entry of the list: its graph and its multiplicity hint.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub synonyms: Vec<String>,
    pub z: Vec<u8>,
    pub charge: Vec<i8>,
    /// `(a, b, order)`, atoms from 0.
    pub bonds: Vec<(usize, usize, u8)>,
    /// `None`: `auto`, the app's automatic multiplicity; `Some(m)`: a forced one.
    pub multiplicity: Option<u8>,
}

impl Entry {
    pub fn net_charge(&self) -> i32 {
        self.charge.iter().map(|&q| q as i32).sum()
    }

    /// Σ Z less the net charge (all electrons; a pseudopotential's core takes an even number, so the
    /// parity is the valence count's too).
    pub fn electrons(&self) -> i32 {
        self.z.iter().map(|&z| z as i32).sum::<i32>() - self.net_charge()
    }

    /// The multiplicity the list states: a forced one as written; `auto` is the app's automatic rule
    /// (`qm::System`: paired, one unpaired electron for an odd count), so 1 for an even electron count
    /// and 2 for an odd one (the radicals).
    pub fn expected_multiplicity(&self) -> u8 {
        self.multiplicity.unwrap_or(if self.electrons() % 2 == 0 { 1 } else { 2 })
    }

    /// The entry drawn as the builder's user would: the atoms in list order, each bond tree grown
    /// breadth-first from atom 0 at the drawn lengths and angles ([`chem::next_direction`],
    /// [`chem::drawn_length`]), then the ring-closing bonds.
    pub fn drawing(&self) -> Drawing {
        let n = self.z.len();
        let mut d = Drawing { atoms: (0..n).map(|i| Atom { z: self.z[i], pos: [0.0; 2], charge: self.charge[i] }).collect(), bonds: Vec::new() };
        let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
        for &(a, b, o) in &self.bonds {
            adj[a].push((b, o));
            adj[b].push((a, o));
        }
        for a in &mut adj {
            a.sort_unstable();
        }
        let mut placed = vec![false; n];
        let mut next_root_x = 0.0f32;
        for root in 0..n {
            if placed[root] {
                continue;
            }
            placed[root] = true;
            d.atoms[root].pos = [next_root_x, 0.0];
            let mut queue = VecDeque::from([root]);
            while let Some(i) = queue.pop_front() {
                for &(j, order) in &adj[i] {
                    if placed[j] {
                        continue;
                    }
                    let angle = chem::next_direction(&d, i, order, None);
                    let len = chem::drawn_length(self.z[i], self.z[j], order);
                    let p = d.atoms[i].pos;
                    d.atoms[j].pos = [p[0] + len * angle.cos(), p[1] + len * angle.sin()];
                    d.bonds.push(Bond { a: i, b: j, order });
                    placed[j] = true;
                    queue.push_back(j);
                }
            }
            next_root_x = d.atoms.iter().map(|a| a.pos[0]).fold(f32::MIN, f32::max) + 3.0;
        }
        for &(a, b, order) in &self.bonds {
            if d.bond_between(a, b).is_none() {
                d.bonds.push(Bond { a, b, order });
            }
        }
        d
    }

    /// The record of this entry with the 3D shape `xyz` (Å, the entry's atom order), rounded to
    /// 1e-5 Å, and its 2D layout ([`principal_layout`]).
    pub fn record(&self, xyz: &[V3], source: Source) -> Record {
        let xyz: Vec<V3> = xyz.iter().map(|p| p.map(|c| (c * 1e5).round() / 1e5 + 0.0)).collect();
        let xy = principal_layout(&xyz, &self.bonds);
        Record {
            name: self.name.clone(),
            synonyms: self.synonyms.clone(),
            source,
            z: self.z.clone(),
            charge: self.charge.clone(),
            bonds: self.bonds.clone(),
            xyz,
            xy,
            stereo_defined: false,
        }
    }

    /// The record has this entry's graph, atom for atom (the generator writes it in list order).
    pub fn same_graph(&self, r: &Record) -> bool {
        let norm = |v: &[(usize, usize, u8)]| {
            let mut v: Vec<_> = v.iter().map(|&(a, b, o)| (a.min(b), a.max(b), o)).collect();
            v.sort_unstable();
            v
        };
        r.z == self.z && r.charge == self.charge && norm(&r.bonds) == norm(&self.bonds)
    }
}

/// Reads the list (its header's format). Blank lines and `#` comments are skipped.
pub fn parse(text: &str) -> Result<Vec<Entry>, String> {
    let mut out: Vec<Entry> = Vec::new();
    for (no, line) in text.lines().enumerate().map(|(i, l)| (i + 1, l.trim())) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |e: String| format!("line {no}: {e}");
        let f: Vec<&str> = line.split('|').map(str::trim).collect();
        let [name, synonyms, atoms, bonds, mult] = f[..] else { return Err(at(format!("{} fields, not 5", f.len()))) };
        if name.is_empty() {
            return Err(at("no name".into()));
        }
        if out.iter().any(|e| e.name == name) {
            return Err(at(format!("{name:?} again")));
        }
        let synonyms: Vec<String> = synonyms.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect();
        let (mut z, mut charge) = (Vec::new(), Vec::new());
        for a in atoms.split_whitespace() {
            let (sym, q) = match a.strip_suffix('+') {
                Some(s) => (s, 1),
                None => a.strip_suffix('-').map_or((a, 0), |s| (s, -1)),
            };
            z.push(z_of_symbol(sym).ok_or_else(|| at(format!("{a:?} is not an element")))?);
            charge.push(q);
        }
        let n = z.len();
        let mut bond_list = Vec::new();
        for b in bonds.split_whitespace() {
            let (sep, order) = [('-', 1u8), ('=', 2), ('#', 3)].into_iter().find(|(c, _)| b.contains(*c)).ok_or_else(|| at(format!("bond {b:?}: no -, = or #")))?;
            let (p, q) = b.split_once(sep).unwrap();
            let (p, q): (usize, usize) = (p.parse().map_err(|_| at(format!("bond {b:?}")))?, q.parse().map_err(|_| at(format!("bond {b:?}")))?);
            if p >= n || q >= n || p == q {
                return Err(at(format!("bond {b:?}: atoms out of range or the same ({n} atoms)")));
            }
            if bond_list.iter().any(|&(a, c, _)| (a, c) == (p, q) || (a, c) == (q, p)) {
                return Err(at(format!("bond {b:?} twice")));
            }
            bond_list.push((p, q, order));
        }
        let multiplicity = match mult {
            "auto" => None,
            m => Some(m.parse::<u8>().ok().filter(|&m| m >= 1).ok_or_else(|| at(format!("multiplicity {m:?}: auto or a number ≥ 1")))?),
        };
        out.push(Entry { name: name.into(), synonyms, z, charge, bonds: bond_list, multiplicity });
    }
    Ok(out)
}

/// The shipped list's entries ([`LIST`]); a list that does not read is a defect (the tests read it).
pub fn entries() -> Vec<Entry> {
    parse(LIST).unwrap_or_else(|e| panic!("lira-molecules.txt: {e}"))
}

/// The 2D layout of a LIRA shape (spec §5.1): the 3D shape projected onto its two largest principal
/// axes, then [`canvas_layout`] (centred, canvas y down, the mean bond 1). When that projection puts
/// two atoms closer than half the mean projected bond (a symmetric top whose principal axes are
/// degenerate, such as tetrahedral BH₄⁻, seen down a bond), the projection plane is instead the one,
/// of 2 000 evenly spread normals, that keeps the closest pair of atoms furthest apart, with its own
/// two principal axes in plane: still a projection of the shape, just a readable one.
pub fn principal_layout(xyz: &[V3], bonds: &[(usize, usize, u8)]) -> Vec<[f32; 2]> {
    let n = xyz.len();
    if n == 0 {
        return Vec::new();
    }
    let c = xyz.iter().fold([0.0; 3], |s, p| [s[0] + p[0], s[1] + p[1], s[2] + p[2]]).map(|s| s / n as f64);
    let p: Vec<V3> = xyz.iter().map(|q| [q[0] - c[0], q[1] - c[1], q[2] - c[2]]).collect();
    let project = |p: &[V3]| -> Vec<V3> {
        let ax = shape::principal_axes(p);
        p.iter().map(|v| [dot(*v, ax[0]), dot(*v, ax[1]), 0.0]).collect()
    };
    let flat = project(&p);
    let q = if readable(&flat, bonds) >= 0.5 {
        flat
    } else {
        const NORMALS: usize = 2000;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let mut best = (f64::NEG_INFINITY, [0.0, 0.0, 1.0]);
        for k in 0..NORMALS {
            let zk = 1.0 - (k as f64 + 0.5) / NORMALS as f64; // the upper hemisphere: ±n is one plane
            let r = (1.0 - zk * zk).sqrt();
            let nk = [r * (golden * k as f64).cos(), r * (golden * k as f64).sin(), zk];
            let onto: Vec<V3> = p.iter().map(|v| sub(*v, scale(nk, dot(*v, nk)))).collect();
            let score = readable(&onto, bonds);
            if score > best.0 + 1e-12 {
                best = (score, nk);
            }
        }
        let nk = best.1;
        project(&p.iter().map(|v| sub(*v, scale(nk, dot(*v, nk)))).collect::<Vec<_>>())
    };
    canvas_layout(&q, bonds)
}

/// The closest pair of atoms of a projection over its mean bond (∞ for one atom; the bare closest
/// distance without bonds).
fn readable(q: &[V3], bonds: &[(usize, usize, u8)]) -> f64 {
    let d = |a: V3, b: V3| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let mean = if bonds.is_empty() { 1.0 } else { bonds.iter().map(|&(a, b, _)| d(q[a], q[b])).sum::<f64>() / bonds.len() as f64 };
    let closest = (0..q.len()).flat_map(|i| (i + 1..q.len()).map(move |j| (i, j))).map(|(i, j)| d(q[i], q[j])).fold(f64::INFINITY, f64::min);
    if mean > 1e-12 { closest / mean } else { 0.0 }
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::iso;

    /// The list reads; every entry is within H–Ar and ≤ 30 atoms, connected, and its drawing is its
    /// graph (it maps onto the entry's own record).
    #[test]
    fn the_lira_list_reads_and_draws_its_graphs() {
        let v = entries();
        assert!(v.len() >= 30, "{} entries", v.len());
        for name in ["hydrogen", "lithium hydride", "borohydride", "trihydrogen cation", "nitric oxide"] {
            assert!(v.iter().any(|e| e.name == name), "{name} is not in the list");
        }
        for e in &v {
            assert!(e.z.len() <= crate::builder::records::MAX_ATOMS && e.z.iter().all(|&z| (1..=18).contains(&z)), "{}", e.name);
            let d = e.drawing();
            assert_eq!(d.components().len(), 1, "{}: not one molecule", e.name);
            // a placeholder shape (the drawing's own embedding, in Å) only to give the record coordinates
            let x: Vec<V3> = crate::builder::embed::embed(&d).iter().map(|p| p.map(units::bohr_to_angstrom)).collect();
            let r = e.record(&x, Source::PubChem { cid: 1 });
            let all: Vec<usize> = (0..d.atoms.len()).collect();
            assert!(matches!(iso::map_fragment(&d, &all, &r), iso::MapResult::Mapped(_)), "{}: its drawing does not map onto its graph", e.name);
            assert!(e.same_graph(&r));
        }
        let bh4 = v.iter().find(|e| e.name == "borohydride").unwrap();
        assert_eq!((bh4.net_charge(), bh4.expected_multiplicity()), (-1, 1));
        let no = v.iter().find(|e| e.name == "nitric oxide").unwrap();
        assert_eq!((no.net_charge(), no.electrons(), no.expected_multiplicity()), (0, 15, 2));
        assert_eq!(v.iter().find(|e| e.name == "trihydrogen cation").unwrap().net_charge(), 1);
    }

    #[test]
    fn the_list_parser_refuses_malformed_lines() {
        for bad in ["a | | H H | 0-1", "a | | Xx | | auto", "a | | H H | 0-2 | auto", "a | | H H | 0-1 1-0 | auto", "a | | H H | 0~1 | auto", "a | | H | | 0", "a | | H | | auto\na | | H | | auto"] {
            assert!(parse(bad).is_err(), "{bad:?} was read");
        }
        let e = &parse("x | p, q | N+ B- | 0#1 | 3").unwrap()[0];
        assert_eq!((e.synonyms.clone(), e.charge.clone(), e.bonds.clone(), e.multiplicity), (vec!["p".to_string(), "q".to_string()], vec![1, -1], vec![(0, 1, 3)], Some(3)));
    }

    /// The layout's projection: a planar shape keeps its plane; a tetrahedron (degenerate principal
    /// axes) seen down a bond is turned so no two atoms overlap; every layout has unit mean bonds.
    #[test]
    fn the_principal_layout_is_a_readable_projection() {
        let y = 0.6 * 3f64.sqrt();
        let tri = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.6, y, 0.0], [-0.6, -y, 0.0]];
        let b3 = [(0, 1, 1), (0, 2, 1), (0, 3, 1)];
        let xy = principal_layout(&tri, &b3);
        let len = |xy: &[[f32; 2]], a: usize, b: usize| ((xy[a][0] - xy[b][0]) as f64).hypot((xy[a][1] - xy[b][1]) as f64);
        for &(a, b, _) in &b3 {
            assert!((len(&xy, a, b) - 1.0).abs() < 1e-5, "{xy:?}");
        }
        // a tetrahedron with one bond exactly along z (the axis a bad projection would look down)
        let (r, rho) = (1.2, 1.2 * 8f64.sqrt() / 3.0);
        let at = |deg: f64| [rho * deg.to_radians().cos(), rho * deg.to_radians().sin(), -r / 3.0];
        let tet = [[0.0, 0.0, 0.0], [0.0, 0.0, r], at(0.0), at(120.0), at(240.0)];
        assert!(readable(&tet.map(|p| [p[0], p[1], 0.0]), &[(0, 1, 1)]) < 1e-9, "the xy plane is the bad projection");
        let b4 = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)];
        let xy = principal_layout(&tet, &b4);
        let closest = (0..5).flat_map(|i| (i + 1..5).map(move |j| (i, j))).map(|(i, j)| len(&xy, i, j)).fold(f64::INFINITY, f64::min);
        assert!(closest > 0.5, "closest pair {closest}: {xy:?}");
        let mean: f64 = b4.iter().map(|&(a, b, _)| len(&xy, a, b)).sum::<f64>() / 4.0;
        assert!((mean - 1.0).abs() < 1e-5);
    }
}
