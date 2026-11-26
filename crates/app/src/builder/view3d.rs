//! The builder's read-only 3D view (spec `docs/superpowers/specs/2026-11-26-builder-3d-view-design.md`):
//! the shape the hand-off would place for Replace scene ([`super::Builder::shaped`]'s positions,
//! Bohr) drawn as a ball-and-stick with the egui painter — no wgpu pass, so the GPU memory budget
//! plays no part and a molecule too big for the box can still be looked at.
//!
//! **Approach.** An orthographic projection after the rotation: the points are turned about their
//! centroid by [`View3d::rot`] into the screen frame (x right, y down, depth into the screen), and
//! scaled so the shape's extent (its farthest atom from the centroid, which no rotation changes)
//! fills [`FIT`] of the rect's smaller side at zoom 1. Depth only orders the drawing back to front
//! ([`draw_order`]) and shades it (farther is dimmer); there is no perspective.
//!
//! **The reset view** ([`View3d::reset`], double-click) is the principal-axes frame of the 2D
//! layouts' projection ([`super::lira::principal_layout`], [`super::records::canvas_layout`]): the
//! largest spread along screen x, the next along screen −y (the canvas's y points down), so a
//! linear molecule lies along x.
//!
//! **Display only (rule 1).** Colours and radii are the render crate's display ones; nothing here
//! reaches the hand-off, `sim`, `qm` or `providers`. Lengths are read out in Å through `units`.
use super::shape::{FragmentSource, Shaped};
use super::{Bond, Drawing};
use egui::{Pos2, Rect, Vec2};
use glam::{DMat3, DQuat, DVec3};

type V3 = [f64; 3];

/// The share of the rect's smaller side the shape's diameter fills at zoom 1.
pub(crate) const FIT: f64 = 0.82;
/// A ball's radius: the render crate's display radius ([`render::elements::radius`], Bohr) times this.
pub(crate) const BALL: f64 = 0.75;
/// Added to the shape's extent before the fit (Bohr): the balls' own size, so the outermost atoms'
/// discs stay inside the rect.
const BALL_MARGIN: f64 = 0.8;
/// The least extent the fit uses (Bohr): one atom, or atoms on one point, are not blown up.
const MIN_EXTENT: f64 = 2.5;
/// The zoom range (the wheel).
pub(crate) const ZOOM: (f64, f64) = (0.2, 8.0);
/// Radians of turn per point of drag.
pub(crate) const TURN_PER_POINT: f64 = 0.01;
/// The most bond angles an atom's readout lists (spec §4).
pub(crate) const MAX_ANGLES: usize = 6;
/// How close (points) the pointer must be to a bond's drawn stick to hover it.
const BOND_PICK: f32 = 5.0;

/// What is drawn and what the pointer can be over: an atom or a bond (indices into the drawing's
/// atoms and bonds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    Atom(usize),
    Bond(usize),
}

/// The view: a rotation (the shape's frame into the screen's) and a zoom. Display state, kept on
/// the builder while its window is closed; never recorded, never handed off.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct View3d {
    pub(crate) rot: DQuat,
    pub(crate) zoom: f64,
    /// The molecule the rotation was set for ([`ViewKey`]): another one resets the view.
    key: Option<ViewKey>,
    /// How many atoms the last frame drew (the UI tests read it).
    pub(crate) shown: usize,
}

impl Default for View3d {
    fn default() -> View3d {
        View3d { rot: DQuat::IDENTITY, zoom: 1.0, key: None, shown: 0 }
    }
}

/// What makes a shape "another molecule" for the view: the drawn graph (elements, charges, bonds)
/// and each fragment's record (approximate ones alike). Moving drawn atoms in 2D, or a lookup
/// timing out, keeps the rotation; a stamp, an erase, Clear or a PubChem answer resets it.
#[derive(Clone, Debug, PartialEq)]
struct ViewKey {
    atoms: Vec<(u8, i8)>,
    bonds: Vec<Bond>,
    records: Vec<(Vec<usize>, Option<FragmentSource>)>,
}

impl ViewKey {
    fn of(d: &Drawing, s: &Shaped) -> ViewKey {
        ViewKey {
            atoms: d.atoms.iter().map(|a| (a.z, a.charge)).collect(),
            bonds: d.bonds.clone(),
            records: s.sources.iter().map(|(f, src)| (f.clone(), (!matches!(src, FragmentSource::Approximate(_))).then(|| src.clone()))).collect(),
        }
    }
}

fn centroid(pos: &[V3]) -> DVec3 {
    if pos.is_empty() {
        return DVec3::ZERO;
    }
    pos.iter().fold(DVec3::ZERO, |s, p| s + DVec3::from(*p)) / pos.len() as f64
}

impl View3d {
    /// The principal-axes view at zoom 1 (identity for fewer than two distinct points).
    pub(crate) fn reset(&mut self, pos: &[V3]) {
        self.zoom = 1.0;
        self.rot = DQuat::IDENTITY;
        let c = centroid(pos);
        let p: Vec<V3> = pos.iter().map(|q| (DVec3::from(*q) - c).to_array()).collect();
        if p.len() < 2 || p.iter().all(|q| DVec3::from(*q).length() < 1e-9) {
            return;
        }
        let [a0, a1, a2] = super::shape::principal_axes(&p).map(DVec3::from);
        // rows a0, −a1, −a2: screen x along the largest spread, screen y (down) along −a1, depth
        // into the screen along −a2 — a proper rotation (a0 × a1 = a2), as seen in the 2D layout
        let m = DMat3::from_cols(a0, -a1, -a2).transpose();
        let q = DQuat::from_mat3(&m).normalize();
        if q.is_finite() {
            self.rot = q;
        }
    }

    /// Resets the view when the shape is another molecule than the one it was set for.
    pub(crate) fn follow(&mut self, d: &Drawing, s: &Shaped) {
        let key = ViewKey::of(d, s);
        if self.key.as_ref() != Some(&key) {
            self.reset(&s.positions);
            self.key = Some(key);
        }
    }

    /// Turns the view by a drag of `delta` points (trackball-style about the centroid: a drag to
    /// the right brings the front of the shape to the right, a drag down brings it down).
    pub(crate) fn turn(&mut self, delta: Vec2) {
        let w = DVec3::new(delta.y as f64, -(delta.x as f64), 0.0) * TURN_PER_POINT;
        if w.length() > 0.0 && w.is_finite() {
            self.rot = (DQuat::from_scaled_axis(w) * self.rot).normalize();
        }
    }

    /// Zooms by the factor `f` within [`ZOOM`].
    pub(crate) fn zoom_by(&mut self, f: f64) {
        if f.is_finite() && f > 0.0 {
            self.zoom = (self.zoom * f).clamp(ZOOM.0, ZOOM.1);
        }
    }

    /// The points turned about their centroid into the screen frame (Bohr: x right, y down, z the
    /// depth, larger is farther).
    pub(crate) fn turned(&self, pos: &[V3]) -> Vec<V3> {
        let c = centroid(pos);
        pos.iter().map(|p| self.rot.mul_vec3(DVec3::from(*p) - c).to_array()).collect()
    }

    /// Points per Bohr in `rect` for these points.
    pub(crate) fn scale(&self, pos: &[V3], rect: Rect) -> f64 {
        let c = centroid(pos);
        let extent = pos.iter().map(|p| (DVec3::from(*p) - c).length()).fold(0.0, f64::max);
        let side = rect.width().min(rect.height()).max(1.0) as f64;
        self.zoom * FIT * side / (2.0 * (extent + BALL_MARGIN).max(MIN_EXTENT))
    }

    /// Each atom's screen centre, depth (Bohr, larger is farther) and ball radius (points); `z`
    /// are the atoms' elements, indexed like `pos`.
    pub(crate) fn project(&self, pos: &[V3], z: &[u8], rect: Rect) -> Vec<(Pos2, f64, f32)> {
        let s = self.scale(pos, rect);
        let o = rect.center();
        self.turned(pos)
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let r = BALL * render::elements::radius(z.get(i).copied().unwrap_or(6)) as f64 * s;
                (Pos2::new(o.x + (t[0] * s) as f32, o.y + (t[1] * s) as f32), t[2], (r as f32).max(1.5))
            })
            .collect()
    }

    /// What the pointer at `p` is over: the nearest (front-most) atom whose ball holds it, else the
    /// front-most bond whose stick passes within a few points.
    pub(crate) fn hit(proj: &[(Pos2, f64, f32)], bonds: &[Bond], p: Pos2) -> Option<Part> {
        let atom = proj.iter().enumerate().filter(|(_, (at, _, r))| at.distance(p) <= r + 2.0).min_by(|x, y| x.1 .1.total_cmp(&y.1 .1)).map(|(i, _)| Part::Atom(i));
        atom.or_else(|| {
            bonds
                .iter()
                .enumerate()
                .filter_map(|(k, b)| {
                    let (a, e) = (proj.get(b.a)?, proj.get(b.b)?);
                    (segment_distance(p, a.0, e.0) <= BOND_PICK).then_some((k, 0.5 * (a.1 + e.1)))
                })
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .map(|(k, _)| Part::Bond(k))
        })
    }
}

fn segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

/// Back to front: every atom and bond, farthest first, so the nearest is drawn last. A bond's depth
/// is its midpoint's; at equal depth a bond goes before an atom (its ends are under the balls).
/// Bonds to atoms that do not exist are left out. One sort per frame.
pub(crate) fn draw_order(proj: &[(Pos2, f64, f32)], bonds: &[Bond]) -> Vec<Part> {
    let mut v: Vec<(f64, u8, Part)> = proj.iter().enumerate().map(|(i, p)| (p.1, 1, Part::Atom(i))).collect();
    v.extend(bonds.iter().enumerate().filter_map(|(k, b)| Some((0.5 * (proj.get(b.a)?.1 + proj.get(b.b)?.1), 0, Part::Bond(k)))));
    v.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    v.into_iter().map(|x| x.2).collect()
}

/// The bond's length in Å (`pos` in Bohr); 0 for a bond to an atom that does not exist.
pub(crate) fn bond_length_angstrom(pos: &[V3], bond: &Bond) -> f64 {
    match (pos.get(bond.a), pos.get(bond.b)) {
        (Some(a), Some(b)) => units::bohr_to_angstrom(DVec3::from(*a).distance(DVec3::from(*b))),
        _ => 0.0,
    }
}

/// The bonded neighbours of `atom` that make an angle at it, in the order of its bonds, each once:
/// a neighbour on the atom itself (Bohr distance under 1e-9) has no direction, so no angle.
fn angle_neighbours(pos: &[V3], bonds: &[Bond], atom: usize) -> Vec<usize> {
    let Some(&c) = pos.get(atom) else { return Vec::new() };
    let c = DVec3::from(c);
    let mut nb: Vec<usize> = Vec::new();
    for j in bonds.iter().filter_map(|b| b.other(atom)) {
        if j < pos.len() && j != atom && !nb.contains(&j) && (DVec3::from(pos[j]) - c).length() >= 1e-9 {
            nb.push(j);
        }
    }
    nb
}

/// The bond angles at `atom` (degrees): one per pair of its bonded neighbours `(j, k)`, j before k
/// in the order of its bonds, at most [`MAX_ANGLES`]. A neighbour on the atom itself has no angle
/// and is left out ([`angle_neighbours`]).
pub(crate) fn angles_at(pos: &[V3], bonds: &[Bond], atom: usize) -> Vec<(usize, usize, f64)> {
    let nb = angle_neighbours(pos, bonds, atom);
    let mut out = Vec::new();
    for (x, &j) in nb.iter().enumerate() {
        for &k in &nb[x + 1..] {
            let c = DVec3::from(pos[atom]);
            let (u, v) = (DVec3::from(pos[j]) - c, DVec3::from(pos[k]) - c);
            out.push((j, k, u.cross(v).length().atan2(u.dot(v)).to_degrees()));
            if out.len() == MAX_ANGLES {
                return out;
            }
        }
    }
    out
}

/// How many angles `atom` has in all, by the same rule as [`angles_at`] before its cap (the
/// readout says how many it left out).
pub(crate) fn angle_count(pos: &[V3], bonds: &[Bond], atom: usize) -> usize {
    let n = angle_neighbours(pos, bonds, atom).len();
    n * n.saturating_sub(1) / 2
}

/// The hover readout of a bond: "C–O 1.208 Å, double".
pub(crate) fn bond_readout(d: &Drawing, pos: &[V3], k: usize) -> Option<String> {
    let b = d.bonds.get(k)?;
    let (za, zb) = (d.atoms.get(b.a)?.z, d.atoms.get(b.b)?.z);
    let order = match b.order {
        2 => ", double",
        3 => ", triple",
        _ => "",
    };
    Some(format!("{}–{} {:.3} Å{order}", render::elements::symbol(za), render::elements::symbol(zb), bond_length_angstrom(pos, b)))
}

/// The hover readout of an atom: its element (and formal charge), then its bond angles.
pub(crate) fn atom_readout(d: &Drawing, pos: &[V3], i: usize) -> Option<String> {
    use render::elements::symbol;
    let a = d.atoms.get(i)?;
    let mut s = format!("{} ({})", symbol(a.z), crate::names::element_name(a.z));
    if a.charge != 0 {
        s += &format!(", charge {:+}", a.charge);
    }
    let angles = angles_at(pos, &d.bonds, i);
    for &(j, k, deg) in &angles {
        s += &format!("\n{}–{}–{} {deg:.1}°", symbol(d.atoms[j].z), symbol(a.z), symbol(d.atoms[k].z));
    }
    let all = angle_count(pos, &d.bonds, i);
    if all > angles.len() {
        s += &format!("\n… {} more", all - angles.len());
    }
    Some(s)
}

/// The atoms of the fragments whose shape is the embedding's guess (drawn with a muted outline,
/// spec §6), as a mask over the drawing's atoms.
pub(crate) fn approximate_atoms(n: usize, s: &Shaped) -> Vec<bool> {
    let mut m = vec![false; n];
    for (frag, src) in &s.sources {
        if matches!(src, FragmentSource::Approximate(_)) {
            for &i in frag {
                if let Some(x) = m.get_mut(i) {
                    *x = true;
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::super::records::Library;
    use super::super::shape::{shape, tests::{draw_record, rec}, LocalOnly};
    use super::*;

    fn dist(p: V3, q: V3) -> f64 {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::new(10.0, 20.0), egui::vec2(800.0, 600.0))
    }

    /// A rotation is a rigid turn about the centroid: every distance is kept.
    #[test]
    fn a_rotation_keeps_distances() {
        let pos = [[0.3, -1.2, 2.0], [1.9, 0.4, -0.7], [-2.2, 1.1, 0.5], [0.0, 0.0, 3.3]];
        let none = View3d::default();
        let turned = View3d { rot: DQuat::from_axis_angle(glam::DVec3::new(1.0, 2.0, -0.5).normalize(), 1.234), zoom: 1.7, ..Default::default() };
        let (a, b) = (none.turned(&pos), turned.turned(&pos));
        assert_ne!(a, b, "the rotation turned something");
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                let want = dist(pos[i], pos[j]);
                assert!((dist(a[i], a[j]) - want).abs() < 1e-12, "{i}, {j} unturned");
                assert!((dist(b[i], b[j]) - want).abs() < 1e-12, "{i}, {j} turned: {} vs {want}", dist(b[i], b[j]));
            }
        }
        // and the projection is those turned points, centred in the rect
        let z = [6, 8, 1, 7];
        let p = turned.project(&pos, &z, rect());
        assert_eq!(p.len(), pos.len());
        for (k, (at, depth, r)) in p.iter().enumerate() {
            assert!(at.x.is_finite() && at.y.is_finite() && depth.is_finite() && *r > 0.0, "{k}: {at:?} {depth} {r}");
            assert!((depth - b[k][2]).abs() < 1e-12, "the depth is the turned z");
        }
    }

    /// The reset view is the principal-axes frame: a linear molecule lies along the screen's x.
    #[test]
    fn the_reset_view_of_a_linear_molecule_lies_along_x() {
        // CO₂ along a tilted line, off the origin
        let dir = glam::DVec3::new(0.3, -0.8, 0.52).normalize();
        let c = glam::DVec3::new(1.0, 2.0, -3.0);
        let pos: Vec<V3> = [-2.2, 0.0, 2.2].iter().map(|&t| (c + dir * t).to_array()).collect();
        let mut v = View3d { rot: DQuat::from_axis_angle(glam::DVec3::Y, 0.9), zoom: 2.5, ..Default::default() };
        v.reset(&pos);
        assert_eq!(v.zoom, 1.0, "the reset zoom");
        let p = v.project(&pos, &[8, 6, 8], rect());
        for (at, _, _) in &p {
            assert!((at.y - p[1].0.y).abs() < 1e-6, "{p:?}");
        }
        assert!((p[0].0.x - p[2].0.x).abs() > 100.0, "spread along x: {p:?}");
        assert!((p[1].0.x - rect().center().x).abs() < 1e-3 && (p[1].0.y - rect().center().y).abs() < 1e-3, "the centroid at the rect's centre: {p:?}");
    }

    /// Back to front: the nearer atom (smaller depth) is drawn last; a bond goes in before the
    /// atoms at its depth.
    #[test]
    fn the_nearer_atom_is_drawn_last() {
        let pos = [[0.0, 0.0, -2.0], [0.5, 0.0, 2.0]];
        let v = View3d::default();
        let p = v.project(&pos, &[6, 6], rect());
        assert!(p[0].1 < p[1].1, "atom 0 is nearer: {p:?}");
        assert_eq!(draw_order(&p, &[]), vec![Part::Atom(1), Part::Atom(0)]);
        // turned half a turn about y, atom 1 is the nearer one
        let back = View3d { rot: DQuat::from_axis_angle(glam::DVec3::Y, std::f64::consts::PI), zoom: 1.0, ..Default::default() };
        let p = back.project(&pos, &[6, 6], rect());
        assert_eq!(draw_order(&p, &[]), vec![Part::Atom(0), Part::Atom(1)]);
        // with a bond between them it lies between the two
        let bond = Bond { a: 0, b: 1, order: 1 };
        assert_eq!(draw_order(&p, &[bond]), vec![Part::Atom(0), Part::Bond(0), Part::Atom(1)]);
    }

    /// Water's shipped PubChem record through the hand-off's shape: the readouts are the record's own
    /// O–H distance (Å) and H–O–H angle.
    #[test]
    fn water_bond_and_angle_read_out_from_the_record() {
        let w = rec("water");
        let mut d = Drawing::default();
        let idx = draw_record(&mut d, w, [0.0, 0.0], &[0, 1, 2]);
        let s = shape(&d, &LocalOnly(Library::shipped()));
        assert!(matches!(s.sources[0].1, FragmentSource::PubChem { .. }), "{:?}", s.sources);
        let o = (0..3).find(|&k| w.z[k] == 8).unwrap();
        let hs: Vec<usize> = (0..3).filter(|&k| w.z[k] == 1).collect();
        for (k, b) in d.bonds.iter().enumerate() {
            let (ra, rb) = ((0..3).find(|&r| idx[r] == b.a).unwrap(), (0..3).find(|&r| idx[r] == b.b).unwrap());
            let want = dist(w.xyz[ra], w.xyz[rb]);
            let got = bond_length_angstrom(&s.positions, b);
            assert!((got - want).abs() < 1e-9, "bond {k}: {got} Å vs the record's {want} Å");
        }
        let (u, v) = (glam::DVec3::from(w.xyz[hs[0]]) - glam::DVec3::from(w.xyz[o]), glam::DVec3::from(w.xyz[hs[1]]) - glam::DVec3::from(w.xyz[o]));
        let want = u.angle_between(v).to_degrees();
        let a = angles_at(&s.positions, &d.bonds, idx[o]);
        assert_eq!(a.len(), 1, "{a:?}");
        let (j, k, deg) = a[0];
        assert_eq!([j.min(k), j.max(k)], [idx[hs[0]].min(idx[hs[1]]), idx[hs[0]].max(idx[hs[1]])]);
        assert!((deg - want).abs() < 1e-9, "H–O–H {deg}° vs the record's {want}°");
        assert!((100.0..110.0).contains(&deg), "{deg}");
        // a hydrogen has one neighbour: no angle
        assert!(angles_at(&s.positions, &d.bonds, idx[hs[0]]).is_empty());
    }

    /// Spec §6: at most six angles at an atom (sulfur hexafluoride's 15 are cut to 6).
    #[test]
    fn at_most_six_angles() {
        let mut pos = vec![[0.0; 3]];
        let mut bonds = Vec::new();
        for (k, a) in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]].into_iter().enumerate() {
            pos.push(a.map(|c| 3.0 * c));
            bonds.push(Bond { a: 0, b: k + 1, order: 1 });
        }
        let a = angles_at(&pos, &bonds, 0);
        assert_eq!(a.len(), 6);
        assert!(a.iter().all(|&(_, _, d)| (d - 90.0).abs() < 1e-9 || (d - 180.0).abs() < 1e-9), "{a:?}");
    }

    /// Review focus: empty, one-atom and coincident drawings give finite views and no angles.
    #[test]
    fn degenerate_drawings_do_not_panic() {
        let cases: [(&[V3], &[u8]); 4] = [
            (&[], &[]),
            (&[[1.0, 2.0, 3.0]], &[6]),
            (&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], &[1, 1]),
            (&[[0.0; 3], [1.4, 0.0, 0.0]], &[1, 1]),
        ];
        for (pos, z) in cases {
            let mut v = View3d::default();
            v.reset(pos);
            assert!(v.rot.is_finite() && v.rot.is_normalized(), "{pos:?}: {:?}", v.rot);
            let p = v.project(pos, z, rect());
            assert_eq!(p.len(), pos.len());
            assert!(p.iter().all(|(at, d, r)| at.x.is_finite() && at.y.is_finite() && d.is_finite() && r.is_finite()), "{pos:?}: {p:?}");
            let bonds: Vec<Bond> = if pos.len() == 2 { vec![Bond { a: 0, b: 1, order: 1 }] } else { vec![] };
            for i in 0..pos.len() {
                assert!(angles_at(pos, &bonds, i).is_empty());
            }
            assert_eq!(draw_order(&p, &bonds).len(), pos.len() + bonds.len());
            // even a zero rect
            let q = v.project(pos, z, Rect::from_min_size(Pos2::ZERO, egui::Vec2::ZERO));
            assert!(q.iter().all(|(at, d, r)| at.x.is_finite() && at.y.is_finite() && d.is_finite() && r.is_finite()));
        }
        // the coincident pair's bond: zero length, no NaN
        assert_eq!(bond_length_angstrom(&[[0.5; 3], [0.5; 3]], &Bond { a: 0, b: 1, order: 1 }), 0.0);
        // three atoms at one point bonded to the middle one: the angle is undefined, so not listed
        let same = [[0.0; 3]; 3];
        assert!(angles_at(&same, &[Bond { a: 1, b: 0, order: 1 }, Bond { a: 1, b: 2, order: 1 }], 1).is_empty());
    }

    /// Water drawn from its shipped record, its hand-off shape, and the record's indices in the
    /// drawing (identity here).
    fn water() -> (Drawing, Shaped) {
        let mut d = Drawing::default();
        draw_record(&mut d, rec("water"), [0.0, 0.0], &[0, 1, 2]);
        let s = shape(&d, &LocalOnly(Library::shipped()));
        (d, s)
    }

    /// He–He: in no database, so the embedding's guess.
    fn helium_pair() -> (Drawing, Shaped) {
        let mut d = Drawing::default();
        d.add_atom(2, [0.0, 0.0]);
        d.add_atom(2, [1.0, 0.0]);
        d.bond(0, 1);
        let s = shape(&d, &LocalOnly(Library::shipped()));
        (d, s)
    }

    /// Spec "Hover": the pointer on an atom's centre is that atom, on a bond's midpoint that bond,
    /// on empty space nothing; inside a ball an atom wins over the bond under it, and of two balls
    /// over one point the front-most (smaller depth) wins, whatever their order.
    #[test]
    fn hover_finds_the_atom_the_bond_or_nothing() {
        let (d, s) = water();
        let z: Vec<u8> = d.atoms.iter().map(|a| a.z).collect();
        let mut v = View3d::default();
        v.reset(&s.positions);
        let p = v.project(&s.positions, &z, rect());
        let hit = |at: Pos2| View3d::hit(&p, &d.bonds, at);
        for i in 0..3 {
            assert_eq!(hit(p[i].0), Some(Part::Atom(i)), "atom {i}");
        }
        for (k, b) in d.bonds.iter().enumerate() {
            let mid = p[b.a].0 + (p[b.b].0 - p[b.a].0) * 0.5;
            assert_eq!(hit(mid), Some(Part::Bond(k)), "bond {k} at {mid:?}: {p:?}");
            // on the bond's line but inside the O ball: the atom
            let o = if z[b.a] == 8 { b.a } else { b.b };
            let h = b.other(o).unwrap();
            let inside = p[o].0 + (p[h].0 - p[o].0).normalized() * (0.8 * p[o].2);
            assert_eq!(hit(inside), Some(Part::Atom(o)), "bond {k}: the ball covers its end");
        }
        assert_eq!(hit(rect().left_top() + Vec2::splat(5.0)), None, "empty space");
        // two balls over one point: the nearer one (depth −1 against +1), in either order
        for (pos, front) in [([[0.0, 0.0, -1.0], [0.1, 0.0, 1.0]], 0), ([[0.1, 0.0, 1.0], [0.0, 0.0, -1.0]], 1)] {
            let q = View3d::default().project(&pos, &[6, 6], rect());
            assert!(q[front].1 < q[1 - front].1);
            assert_eq!(View3d::hit(&q, &[], q[1 - front].0), Some(Part::Atom(front)), "{pos:?}");
        }
    }

    /// The hover readouts: a bond's length in Å (the record's), an atom's element and bond angles.
    #[test]
    fn the_readouts_say_the_length_and_the_angles() {
        let (d, s) = water();
        let w = rec("water");
        for (k, b) in d.bonds.iter().enumerate() {
            let t = bond_readout(&d, &s.positions, k).unwrap();
            let want = dist(w.xyz[b.a], w.xyz[b.b]);
            assert!(t.ends_with(" Å") && t.contains(&format!("{want:.3}")) && (t.starts_with("O–H") || t.starts_with("H–O")), "{t} vs {want}");
        }
        assert_eq!(bond_readout(&d, &s.positions, 9), None);
        let o = d.atoms.iter().position(|a| a.z == 8).unwrap();
        let t = atom_readout(&d, &s.positions, o).unwrap();
        assert!(t.starts_with("O (oxygen)") && t.contains("H–O–H") && t.contains('°') && !t.contains("more"), "{t}");
        // a double bond says so; a seventh angle is counted, not listed
        let mut c = Drawing::default();
        c.add_atom(6, [0.0, 0.0]);
        c.add_atom(8, [1.0, 0.0]);
        c.bond(0, 1);
        c.cycle_order(0);
        assert!(bond_readout(&c, &[[0.0; 3], [2.3, 0.0, 0.0]], 0).unwrap().ends_with(", double"));
        let mut sf = Drawing::default();
        sf.add_atom(16, [0.0, 0.0]);
        let mut pos = vec![[0.0; 3]];
        for (k, a) in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]].into_iter().enumerate() {
            sf.add_atom(9, [k as f32, 1.0]);
            sf.bond(0, k + 1);
            pos.push(a.map(|x| 3.0 * x));
        }
        let t = atom_readout(&sf, &pos, 0).unwrap();
        assert_eq!(t.matches('°').count(), 6, "{t}");
        assert!(t.ends_with("… 9 more"), "15 angles in all: {t}");
        // a neighbour on the atom itself is no angle, nor counted (review: the same rule as angles_at)
        pos[6] = [0.0; 3];
        assert_eq!(angle_count(&pos, &sf.bonds, 0), 10);
        assert!(atom_readout(&sf, &pos, 0).unwrap().ends_with("… 4 more"));
    }

    /// Spec §6: the guessed fragment's atoms are marked, a database shape's are not.
    #[test]
    fn approximate_atoms_are_the_guessed_fragments() {
        let (d, s) = helium_pair();
        assert!(matches!(s.sources[0].1, FragmentSource::Approximate(_)), "{:?}", s.sources);
        assert_eq!(approximate_atoms(d.atoms.len(), &s), vec![true, true]);
        let (d, s) = water();
        assert_eq!(approximate_atoms(d.atoms.len(), &s), vec![false, false, false]);
    }

    /// Review focus: the rotation is kept while the molecule is the same one (a 2D atom move), and
    /// reset for another (a bond or an element changed).
    #[test]
    fn the_view_follows_the_molecule_not_the_sketch() {
        let (d, s) = water();
        let mut v = View3d::default();
        v.follow(&d, &s);
        let mut fresh = View3d::default();
        fresh.reset(&s.positions);
        assert_eq!(v.rot, fresh.rot, "first sight: the reset view");
        v.turn(Vec2::new(80.0, -30.0));
        let turned = v.rot;
        assert!(turned.angle_between(fresh.rot) > 0.5);
        // an atom moved on the 2D canvas: the same molecule
        let mut moved = d.clone();
        moved.atoms[1].pos[0] += 0.7;
        let s2 = shape(&moved, &LocalOnly(Library::shipped()));
        v.follow(&moved, &s2);
        assert_eq!(v.rot, turned, "kept");
        // another element, or another bond: reset
        for change in [0, 1] {
            let mut other = d.clone();
            if change == 0 {
                other.set_z(0, 16);
            } else {
                other.remove_bond(0);
            }
            let s3 = shape(&other, &LocalOnly(Library::shipped()));
            let mut v = v.clone();
            v.follow(&other, &s3);
            let mut want = View3d::default();
            want.reset(&s3.positions);
            assert_eq!(v.rot, want.rot, "change {change}: reset");
        }
    }
}
