//! 2D → 3D (spec §5): the drawing's starting geometry in Bohr. A pure function, unit-tested.
//!
//! An editor's embedding, not physics (spec §3): bond springs to r₀ (covalent radii × the order
//! factor, [`chem`]), springs on the 1–3 distances to the VSEPR angle of the centre (steric number
//! SN = σ bonds + lone pairs: 180°, 120°, 109.47° — 107° / 104.5° with one / two lone pairs), a soft
//! repulsion between atoms further apart in the graph, and a planarity term for trigonal centres
//! and across double bonds (ethylene and benzene stay flat). It starts from the drawing scaled to
//! the bond lengths with small deterministic out-of-plane offsets, so a tetrahedral centre can leave
//! the plane, and relaxes by gradient descent. The result is only the guess the solver's Relax
//! starts from.
use super::chem;
use super::Drawing;

type V3 = [f64; 3];

/// The 1–3 distance spring's stiffness relative to the bonds'.
const K_BOND: f64 = 1.0;
const K_ANGLE: f64 = 0.4;
const K_REPEL: f64 = 0.5;
const K_PLANE: f64 = 2.0;
/// The repulsion acts below max(1.2 × the radius sum, this) — Bohr.
const REPEL_FLOOR: f64 = 3.0;
/// The triple product's length scale (Bohr): V / s³ is dimensionless.
const PLANE_SCALE: f64 = 2.5;
/// Out-of-plane start offsets: ± this (Bohr).
const Z_JITTER: f64 = 0.3;
/// The descent's step cap per atom (Bohr) and its budget.
const STEP_CAP: f64 = 0.2;
pub const MAX_ITERATIONS: usize = 2000;
/// Converged when no atom feels a gradient above this.
const GRAD_TOL: f64 = 1e-6;
/// Bohr per canvas unit when nothing is bonded (a C–C single bond).
const UNBONDED_SCALE: f64 = 2.9;

/// One distance spring: atoms, target, stiffness, and which side it acts on.
#[derive(Clone, Copy)]
struct Spring {
    i: usize,
    j: usize,
    d0: f64,
    k: f64,
    /// Only pushes apart (d < d0): the repulsion, and the 1–3 minimum of a centre with SN ≥ 5.
    push_only: bool,
}

/// A planarity term: centre `c` and three atoms whose vectors from it must be coplanar.
#[derive(Clone, Copy)]
struct Plane {
    c: usize,
    a: usize,
    b: usize,
    d: usize,
}

/// The ideal angle at a centre (degrees) from its steric number and lone pairs; `None` when the
/// centre has no single ideal angle (SN ≥ 5: only kept from crowding, at 90°) or too few bonds.
fn ideal_angle(sn: u32, lp: u32) -> Option<f64> {
    match sn {
        2 => Some(180.0),
        3 => Some(120.0),
        4 => Some(match lp {
            0 => 109.47,
            1 => 107.0,
            _ => 104.5,
        }),
        _ => None,
    }
}

/// The terms of a drawing's embedding energy.
struct Terms {
    springs: Vec<Spring>,
    planes: Vec<Plane>,
}

/// The bonds of six-membered rings drawn as alternating single and double bonds (a Kekulé
/// structure): one aromatic ring, whose six bonds are equal — each gets the mean of the single
/// and double lengths ([`AROMATIC_FACTOR`]) instead of alternating 1.52 / 1.32 Å.
fn aromatic_bonds(d: &Drawing) -> Vec<bool> {
    let mut out = vec![false; d.bonds.len()];
    let order = |a: usize, b: usize| d.bond_between(a, b).map(|k| d.bonds[k].order);
    // every simple 6-cycle through its lowest atom
    fn walk(d: &Drawing, path: &mut Vec<usize>, rings: &mut Vec<Vec<usize>>) {
        let last = *path.last().unwrap();
        for j in d.neighbours(last) {
            if path.len() == 6 {
                if j == path[0] {
                    rings.push(path.clone());
                }
            } else if j > path[0] && !path.contains(&j) {
                path.push(j);
                walk(d, path, rings);
                path.pop();
            }
        }
    }
    let mut rings = Vec::new();
    for s in 0..d.atoms.len() {
        walk(d, &mut vec![s], &mut rings);
    }
    for ring in rings {
        let orders: Vec<Option<u8>> = (0..6).map(|k| order(ring[k], ring[(k + 1) % 6])).collect();
        let alternates = orders.iter().enumerate().all(|(k, o)| *o == Some(if k % 2 == 0 { orders[0].unwrap_or(0) } else { 3 - orders[0].unwrap_or(0) }));
        if alternates && matches!(orders[0], Some(1 | 2)) {
            for k in 0..6 {
                if let Some(b) = d.bond_between(ring[k], ring[(k + 1) % 6]) {
                    out[b] = true;
                }
            }
        }
    }
    out
}

/// An aromatic bond's length factor: the mean of the single (1) and double (0.87) factors of
/// spec §3 (C–C 1.42 Å; benzene's is 1.39 Å — the Relax corrects the rest).
const AROMATIC_FACTOR: f64 = (1.0 + 0.87) / 2.0;

fn terms(d: &Drawing) -> Terms {
    let n = d.atoms.len();
    let mut springs = Vec::new();
    let mut planes = Vec::new();
    let mut near = vec![vec![false; n]; n];
    let aromatic = aromatic_bonds(d);
    let r0 = |a: usize, b: usize| match d.bond_between(a, b) {
        Some(k) if aromatic[k] => (chem::covalent_radius_bohr(d.atoms[a].z) + chem::covalent_radius_bohr(d.atoms[b].z)) * AROMATIC_FACTOR,
        Some(k) => chem::bond_length(d.atoms[a].z, d.atoms[b].z, d.bonds[k].order),
        None => chem::bond_length(d.atoms[a].z, d.atoms[b].z, 1),
    };
    for b in &d.bonds {
        springs.push(Spring { i: b.a, j: b.b, d0: r0(b.a, b.b), k: K_BOND, push_only: false });
        near[b.a][b.b] = true;
        near[b.b][b.a] = true;
    }
    for c in 0..n {
        let nb = d.neighbours(c);
        let a = &d.atoms[c];
        let lp = chem::lone_pairs(a.z, a.charge, d.bond_sum(c));
        let sn = nb.len() as u32 + lp;
        let angle = ideal_angle(sn, lp);
        for x in 0..nb.len() {
            for y in x + 1..nb.len() {
                let (i, k) = (nb[x], nb[y]);
                let (ri, rk) = (r0(c, i), r0(c, k));
                let theta = angle.unwrap_or(90.0).to_radians();
                let d13 = (ri * ri + rk * rk - 2.0 * ri * rk * theta.cos()).max(0.0).sqrt();
                springs.push(Spring { i, j: k, d0: d13, k: K_ANGLE, push_only: angle.is_none() });
                near[i][k] = true;
                near[k][i] = true;
            }
        }
        // a trigonal centre with three σ neighbours stays in their plane
        if nb.len() == 3 && sn == 3 {
            planes.push(Plane { c, a: nb[0], b: nb[1], d: nb[2] });
        }
    }
    // across a double bond: every substituent of one end in the plane of the other end's
    for bd in d.bonds.iter().filter(|b| b.order == 2) {
        for k in d.neighbours(bd.a).into_iter().filter(|&k| k != bd.b) {
            for l in d.neighbours(bd.b).into_iter().filter(|&l| l != bd.a) {
                planes.push(Plane { c: bd.a, a: k, b: bd.b, d: l });
            }
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            if !near[i][j] && i != j {
                let reach = 1.2 * (chem::covalent_radius_bohr(d.atoms[i].z) + chem::covalent_radius_bohr(d.atoms[j].z));
                springs.push(Spring { i, j, d0: reach.max(REPEL_FLOOR), k: K_REPEL, push_only: true });
            }
        }
    }
    Terms { springs, planes }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// The energy and its gradient at `x`.
fn energy(t: &Terms, x: &[V3], g: &mut [V3]) -> f64 {
    g.iter_mut().for_each(|v| *v = [0.0; 3]);
    let mut e = 0.0;
    for s in &t.springs {
        let r = sub(x[s.i], x[s.j]);
        let d = norm(r).max(1e-9);
        let diff = d - s.d0;
        if s.push_only && diff >= 0.0 {
            continue;
        }
        e += 0.5 * s.k * diff * diff;
        let f = s.k * diff / d;
        for c in 0..3 {
            g[s.i][c] += f * r[c];
            g[s.j][c] -= f * r[c];
        }
    }
    let s6 = PLANE_SCALE.powi(6);
    for p in &t.planes {
        let (u, v, w) = (sub(x[p.a], x[p.c]), sub(x[p.b], x[p.c]), sub(x[p.d], x[p.c]));
        let vol = dot(u, cross(v, w));
        e += 0.5 * K_PLANE * vol * vol / s6;
        let f = K_PLANE * vol / s6;
        let (ga, gb, gd) = (cross(v, w), cross(w, u), cross(u, v));
        for c in 0..3 {
            g[p.a][c] += f * ga[c];
            g[p.b][c] += f * gb[c];
            g[p.d][c] += f * gd[c];
            g[p.c][c] -= f * (ga[c] + gb[c] + gd[c]);
        }
    }
    e
}

/// The deterministic out-of-plane start offset of atom `i` (a hash of the index), in ± [`Z_JITTER`].
fn jitter(i: usize) -> f64 {
    let h = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    ((h >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * Z_JITTER
}

/// The drawing's atoms in 3D (Bohr), centred at the origin (spec §5).
pub fn embed(d: &Drawing) -> Vec<V3> {
    let n = d.atoms.len();
    if n == 0 {
        return Vec::new();
    }
    // the drawing scaled so its mean bond is the mean r₀; canvas y points down, the scene's y up
    let drawn: Vec<f64> = d
        .bonds
        .iter()
        .map(|b| {
            let (p, q) = (d.atoms[b.a].pos, d.atoms[b.b].pos);
            (((p[0] - q[0]) as f64).powi(2) + ((p[1] - q[1]) as f64).powi(2)).sqrt()
        })
        .collect();
    let mean_drawn = drawn.iter().sum::<f64>() / drawn.len().max(1) as f64;
    let mean_r0 = d.bonds.iter().map(|b| chem::bond_length(d.atoms[b.a].z, d.atoms[b.b].z, b.order)).sum::<f64>() / d.bonds.len().max(1) as f64;
    let scale = if d.bonds.is_empty() || mean_drawn < 1e-3 { UNBONDED_SCALE } else { mean_r0 / mean_drawn };
    let mut x: Vec<V3> = d.atoms.iter().enumerate().map(|(i, a)| [a.pos[0] as f64 * scale, -(a.pos[1] as f64) * scale, jitter(i)]).collect();
    let t = terms(d);
    let mut g = vec![[0.0; 3]; n];
    let mut trial = x.clone();
    let mut gt = g.clone();
    let mut e = energy(&t, &x, &mut g);
    let mut alpha = 0.1;
    for _ in 0..MAX_ITERATIONS {
        let gmax = g.iter().map(|v| norm(*v)).fold(0.0, f64::max);
        if gmax < GRAD_TOL {
            break;
        }
        // a backtracking step: halve until the energy drops, then grow
        loop {
            let cap = (alpha * gmax).max(1e-300);
            let shrink = if cap > STEP_CAP { STEP_CAP / cap } else { 1.0 };
            for i in 0..n {
                for c in 0..3 {
                    trial[i][c] = x[i][c] - alpha * shrink * g[i][c];
                }
            }
            let et = energy(&t, &trial, &mut gt);
            if et <= e {
                std::mem::swap(&mut x, &mut trial);
                std::mem::swap(&mut g, &mut gt);
                e = et;
                alpha = (alpha * 1.3).min(10.0);
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-12 {
                break;
            }
        }
        if alpha < 1e-12 {
            break;
        }
    }
    centre(&mut x);
    x
}

/// Moves the centroid to the origin.
pub fn centre(x: &mut [V3]) {
    if x.is_empty() {
        return;
    }
    let mut c = [0.0; 3];
    for p in x.iter() {
        for k in 0..3 {
            c[k] += p[k] / x.len() as f64;
        }
    }
    for p in x.iter_mut() {
        for k in 0..3 {
            p[k] -= c[k];
        }
    }
}

/// The closest approach between two sets of points (∞ if either is empty).
pub fn closest(a: &[V3], b: &[V3]) -> f64 {
    a.iter().flat_map(|p| b.iter().map(move |q| norm(sub(*p, *q)))).fold(f64::INFINITY, f64::min)
}

/// The gap Insert keeps between the new molecule and the scene's atoms (Bohr, spec §5 step 4).
pub const BESIDE_GAP: f64 = 3.0;

/// Insert's placement (spec §5 step 4): `new` (centred at the origin) moved along +x in 0.25 Bohr
/// steps until it is beside `existing` — its centroid past their largest x — with every pair at
/// least [`BESIDE_GAP`] apart. Nothing moves when the scene is empty.
pub fn place_beside(existing: &[V3], new: &mut [V3]) {
    if existing.is_empty() || new.is_empty() {
        return;
    }
    let max_x = existing.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    let cx = |v: &[V3]| v.iter().map(|p| p[0]).sum::<f64>() / v.len() as f64;
    for _ in 0..100_000 {
        if cx(new) > max_x && closest(existing, new) >= BESIDE_GAP {
            return;
        }
        for p in new.iter_mut() {
            p[0] += 0.25;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::chem::{fill_hydrogens, stamp, Stamp};
    use super::*;

    fn angle(x: &[V3], a: usize, c: usize, b: usize) -> f64 {
        let (u, v) = (sub(x[a], x[c]), sub(x[b], x[c]));
        (dot(u, v) / (norm(u) * norm(v))).clamp(-1.0, 1.0).acos().to_degrees()
    }

    /// The largest distance of the atoms from their best-fit plane (the smallest eigenvalue's
    /// direction of the covariance, by power iteration on the shifted matrix).
    fn out_of_plane(x: &[V3]) -> f64 {
        let mut y = x.to_vec();
        centre(&mut y);
        let mut m = [[0.0; 3]; 3];
        for p in &y {
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += p[i] * p[j];
                }
            }
        }
        let tr = m[0][0] + m[1][1] + m[2][2];
        // the largest eigenvector of (tr I − m) is the smallest of m: the plane's normal
        let s = |v: V3| {
            let mut w = [0.0; 3];
            for i in 0..3 {
                w[i] = tr * v[i] - (0..3).map(|j| m[i][j] * v[j]).sum::<f64>();
            }
            let n = norm(w);
            [w[0] / n, w[1] / n, w[2] / n]
        };
        let mut nrm = [0.3, 0.5, 0.8];
        for _ in 0..500 {
            nrm = s(nrm);
        }
        y.iter().map(|p| dot(*p, nrm).abs()).fold(0.0, f64::max)
    }

    fn from(z: &[u8], bonds: &[(usize, usize, u8)]) -> Drawing {
        let mut d = Drawing::default();
        for (i, &zz) in z.iter().enumerate() {
            d.add_atom(zz, [i as f32 * 0.9, (i % 2) as f32 * 0.5]);
        }
        for &(a, b, o) in bonds {
            d.bond(a, b);
            let k = d.bond_between(a, b).unwrap();
            d.bonds[k].order = o;
        }
        d
    }

    fn lone(z: u8) -> Drawing {
        let mut d = Drawing::default();
        d.add_atom(z, [0.0, 0.0]);
        fill_hydrogens(&mut d);
        d
    }

    fn min_nonbonded(d: &Drawing, x: &[V3]) -> f64 {
        let mut m = f64::INFINITY;
        for i in 0..x.len() {
            for j in i + 1..x.len() {
                if d.bond_between(i, j).is_none() {
                    m = m.min(norm(sub(x[i], x[j])));
                }
            }
        }
        m
    }

    #[test]
    fn ch4_is_tetrahedral() {
        let d = lone(6);
        let x = embed(&d);
        for a in 1..5 {
            for b in a + 1..5 {
                let t = angle(&x, a, 0, b);
                assert!((t - 109.47).abs() < 3.0, "H–C–H {t}");
            }
            let r = norm(sub(x[a], x[0]));
            assert!((r - chem::bond_length(6, 1, 1)).abs() < 0.05, "C–H {r}");
        }
    }

    #[test]
    fn ethylene_is_planar_and_trigonal() {
        let mut d = Drawing::default();
        d.add_atom(6, [0.0, 0.0]);
        d.add_atom(6, [1.0, 0.0]);
        d.bond(0, 1);
        d.cycle_order(0);
        fill_hydrogens(&mut d);
        assert_eq!(d.atoms.len(), 6);
        let x = embed(&d);
        assert!(out_of_plane(&x) < 0.05, "out of plane {}", out_of_plane(&x));
        for c in [0, 1] {
            let nb = d.neighbours(c);
            for p in 0..3 {
                for q in p + 1..3 {
                    let t = angle(&x, nb[p], c, nb[q]);
                    assert!((t - 120.0).abs() < 3.0, "angle at C{c}: {t}");
                }
            }
        }
    }

    #[test]
    fn water_is_bent() {
        let d = lone(8);
        let x = embed(&d);
        let t = angle(&x, 1, 0, 2);
        assert!((t - 104.5).abs() < 3.0, "H–O–H {t}");
        // and NH3 is a pyramid, not flat
        let n = lone(7);
        let x = embed(&n);
        let sum: f64 = [(1, 2), (1, 3), (2, 3)].iter().map(|&(a, b)| angle(&x, a, 0, b)).sum();
        assert!(sum < 340.0, "angles sum to {sum}: a flat NH₃");
    }

    #[test]
    fn benzene_is_planar_with_equal_cc() {
        let mut d = Drawing::default();
        stamp(&mut d, Stamp::Benzene, None, [0.0, 0.0]);
        fill_hydrogens(&mut d);
        let x = embed(&d);
        assert!(out_of_plane(&x) < 0.05, "out of plane {}", out_of_plane(&x));
        let cc: Vec<f64> = d.bonds.iter().filter(|b| d.atoms[b.a].z == 6 && d.atoms[b.b].z == 6).map(|b| norm(sub(x[b.a], x[b.b]))).collect();
        assert_eq!(cc.len(), 6);
        let mean = cc.iter().sum::<f64>() / 6.0;
        for r in &cc {
            assert!((r / mean - 1.0).abs() < 0.02, "C–C {r} vs mean {mean}");
        }
        // only a ring drawn alternating is aromatic: cyclohexane's single bonds and a ring with two
        // adjacent double bonds are not
        let mut hexane = Drawing::default();
        stamp(&mut hexane, Stamp::Benzene, None, [0.0, 0.0]);
        hexane.bonds.iter_mut().for_each(|b| b.order = 1);
        assert!(aromatic_bonds(&hexane).iter().all(|&a| !a));
        hexane.bonds[0].order = 2;
        hexane.bonds[1].order = 2;
        assert!(aromatic_bonds(&hexane).iter().all(|&a| !a));
        assert!(aromatic_bonds(&d).iter().filter(|&&a| a).count() == 6);
    }

    #[test]
    fn no_nonbonded_pair_closer_than_1_5_bohr() {
        // ethanol, acetic acid, benzene, and a crowded drawing (atoms drawn on top of each other)
        let mut ethanol = from(&[6, 6, 8], &[(0, 1, 1), (1, 2, 1)]);
        fill_hydrogens(&mut ethanol);
        let mut acid = from(&[6], &[]);
        stamp(&mut acid, Stamp::Carboxyl, Some(0), [0.0, 0.0]);
        fill_hydrogens(&mut acid);
        let mut benzene = Drawing::default();
        stamp(&mut benzene, Stamp::Benzene, None, [0.0, 0.0]);
        fill_hydrogens(&mut benzene);
        let mut crowded = Drawing::default();
        for k in 0..4 {
            crowded.add_atom(6, [0.01 * k as f32, 0.0]);
        }
        fill_hydrogens(&mut crowded);
        for (name, d) in [("ethanol", ethanol), ("acetic acid", acid), ("benzene", benzene), ("crowded", crowded)] {
            let x = embed(&d);
            assert!(x.iter().all(|p| p.iter().all(|c| c.is_finite())), "{name}");
            let m = min_nonbonded(&d, &x);
            assert!(m >= 1.5, "{name}: a non-bonded pair at {m} Bohr");
        }
    }

    #[test]
    fn embedding_is_deterministic() {
        let mut d = from(&[6, 6, 8], &[(0, 1, 1), (1, 2, 1)]);
        fill_hydrogens(&mut d);
        assert_eq!(embed(&d), embed(&d));
        // a lone atom sits at the origin; nothing drawn is nothing
        assert_eq!(embed(&from(&[2], &[])), vec![[0.0; 3]]);
        assert!(embed(&Drawing::default()).is_empty());
        // centred
        let x = embed(&d);
        let c: f64 = x.iter().map(|p| p[0]).sum::<f64>() / x.len() as f64;
        assert!(c.abs() < 1e-9);
    }

    #[test]
    fn place_beside_keeps_3_bohr() {
        let existing = vec![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let mut d = Drawing::default();
        stamp(&mut d, Stamp::Benzene, None, [0.0, 0.0]);
        fill_hydrogens(&mut d);
        let mut new = embed(&d);
        place_beside(&existing, &mut new);
        assert!(closest(&existing, &new) >= BESIDE_GAP);
        // a quarter step back would be too close or not beside: the smallest shift
        let mut back = new.clone();
        back.iter_mut().for_each(|p| p[0] -= 0.25);
        let cx = back.iter().map(|p| p[0]).sum::<f64>() / back.len() as f64;
        assert!(closest(&existing, &back) < BESIDE_GAP || cx <= 1.4);
        // an empty scene: unmoved
        let mut same = embed(&d);
        let before = same.clone();
        place_beside(&[], &mut same);
        assert_eq!(same, before);
    }
}
