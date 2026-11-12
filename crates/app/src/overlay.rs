//! Reference geometry for the 3D view: the simulation cell, a floor grid with axes (Blender-style,
//! the xz plane with +y up), in Place mode the placement plane the next click lands on, and the
//! bonds read off the density (M6 spec §3.4: [`bond_lines`]). Pure: turns the display toggles and
//! the bond graph into line vertices; the painter draws them depth-tested, so a bond shows between
//! the spheres' surfaces. A bond's line encodes its strength (user request, M6): width and opacity
//! follow log ρ_b ([`strength_fraction`]), the pattern its QTAIM kind (`sim::BondKind`: solid
//! shared, dashed closed-shell, dotted indeterminate), the colour a multi-centre ring. Since M7
//! Task 13 an arc and a degree label sit at each atom between every pair of its drawn bonds
//! ([`bond_angles`], [`angle_arc_lines`]): pure geometry from the nuclear positions, styled after
//! the weaker of the two bonds ([`angle_style`]).
use crate::scene::Scene;
use crate::ui::{ToolMode, UiState};
use glam::Vec3;
use render::camera::OrbitCamera;
use render::lines::{self, LineVertex};

const CELL: [f32; 4] = [0.95, 0.80, 0.35, 0.55];
/// The container walls' face: faint, cooler than the cell.
const WALLS: [f32; 4] = [0.55, 0.85, 1.0, 0.30];
const FLOOR_MINOR: [f32; 4] = [1.0, 1.0, 1.0, 0.09];
const FLOOR_MAJOR: [f32; 4] = [1.0, 1.0, 1.0, 0.22];
const PLACE_MINOR: [f32; 4] = [0.45, 0.80, 1.0, 0.16];
const PLACE_MAJOR: [f32; 4] = [0.45, 0.80, 1.0, 0.40];
/// Major line every this many minor spacings.
const MAJOR_EVERY: i32 = 5;
/// Half-size of the placement-plane grid (Bohr): big enough to frame a molecule, small enough
/// not to read as a floor.
const PLACE_HALF_EXTENT: f32 = 6.0;

/// A bond's colour (warm white) and a multi-centre ring bond's (violet: the three-centre bond of
/// H₃⁺ is one bond shared by three nuclei, not three ordinary ones).
pub const BOND: [f32; 3] = [0.93, 0.91, 0.84];
const RING_BOND: [f32; 3] = [0.74, 0.62, 1.0];
/// Alpha of a bond from the current evaluation, and of one from an earlier DFT step drawn between
/// the current positions (a surrogate step has no electrons: the faded graph, spec §3.4).
pub const BOND_ALPHA: f32 = 0.9;
pub const STALE_BOND_ALPHA: f32 = 0.3;
/// A bond is drawn as a bundle of this many thin lines on a cylinder of this radius (Bohr) at full
/// strength: one pixel-wide line reads as a hairline next to the spheres (display choice, not
/// physics).
pub const BOND_STRANDS: usize = 6;
pub const BOND_RADIUS: f32 = 0.045;
/// The display scale of bond strength — **a display scale, not physics**: log ρ_b mapped from the
/// detector's floor `sim::bonds::RHO_FLOOR` (0.01 e/Bohr³, nothing weaker is a bond) to
/// `RHO_STRONG` = 0.3 e/Bohr³ (the upper end of covalent ρ_b at a bond point — H₂'s is ≈ 0.27
/// near the exact density; this solver's at the preset 1.45 Bohr is 0.232), clamped.
pub const RHO_STRONG: f64 = 0.3;
/// At the floor a bond is drawn at these fractions of the full radius and alpha.
pub const MIN_WIDTH: f32 = 0.3;
pub const MIN_ALPHA: f32 = 0.35;
/// A closed-shell contact's dash and gap (Bohr), and an indeterminate one's dot and gap.
const DASH: f32 = 0.22;
const GAP: f32 = 0.14;
const DOT: f32 = 0.05;
const DOT_GAP: f32 = 0.12;
/// A stale graph's dash and gap (Bohr; user request 2026-11-13): the bonds of an earlier DFT step
/// are broken into long dashes as well as faded, so they read as "not this geometry's" even where
/// the fade is hard to see. The dash holds two closed-shell dashes or four dots whole (2 × (0.22 +
/// 0.14) − 0.14 = 0.58; four dots end at 3 × 0.17 + 0.05 = 0.56), and the gap is almost twice
/// either kind's own, so a stale bond's kind still shows inside its dashes; a gap sits on the
/// bond's middle, so even a short bond between two spheres shows the break. Display only.
pub const STALE_DASH: f32 = 0.58;
pub const STALE_GAP: f32 = 0.26;

/// Where ρ_b sits on the display scale: ln(ρ_b/ρ_floor)/ln(ρ_strong/ρ_floor), clamped to [0, 1]
/// (0 for a non-finite ρ_b).
pub fn strength_fraction(rho_b: f64) -> f32 {
    let f = (rho_b / sim::bonds::RHO_FLOOR).ln() / (RHO_STRONG / sim::bonds::RHO_FLOOR).ln();
    if f.is_finite() { f.clamp(0.0, 1.0) as f32 } else { 0.0 }
}

/// The dash pattern of a kind (Bohr; `None` = solid).
fn pattern(kind: sim::BondKind) -> Option<(f32, f32)> {
    match kind {
        sim::BondKind::Shared => None,
        sim::BondKind::ClosedShell => Some((DASH, GAP)),
        sim::BondKind::Indeterminate => Some((DOT, DOT_GAP)),
    }
}

/// The drawn pieces (start, end) along a line of length `len` (Bohr) with the dash pattern `kind`
/// ([`pattern`]; `None` solid): the kind's pattern from the start, or when `stale` the long
/// [`STALE_DASH`]es with a [`STALE_GAP`] centred on the middle, the kind's pattern laid afresh in
/// each dash (so a stale solid line is long dashes, a stale dashed or dotted one groups of its
/// dashes or dots). Pieces shorter than 1e-4 Bohr are left out.
pub fn dash_pieces(len: f32, kind: Option<(f32, f32)>, stale: bool) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    if !(len > 0.0) {
        return out;
    }
    let lay = |out: &mut Vec<(f32, f32)>, s0: f32, s1: f32| match kind {
        None => out.push((s0, s1)),
        Some((on, gap)) => {
            let mut s = s0;
            while s < s1 {
                let e = (s + on).min(s1);
                out.push((s, e));
                s = e + gap;
            }
        }
    };
    if !stale {
        lay(&mut out, 0.0, len);
    } else {
        // a gap centred on the middle: the dash before it ends at len/2 − gap/2, the others one
        // period apart from there, back to the first one that reaches past the start
        let period = STALE_DASH + STALE_GAP;
        let first_end = 0.5 * (len - STALE_GAP);
        let mut s = first_end - STALE_DASH - period * (first_end / period).floor();
        while s < len {
            let (a, b) = (s.max(0.0), (s + STALE_DASH).min(len));
            if b > a {
                // the kind's pattern from the dash's own start (a clipped first dash: from 0)
                lay(&mut out, a, b);
            }
            s += period;
        }
    }
    out.retain(|&(a, b)| b - a > 1e-4);
    if out.is_empty() {
        // a line shorter than the centred gap (a tiny arc): its kind's pattern, never nothing
        lay(&mut out, 0.0, len);
    }
    out
}

/// How a drawn bond is styled: from the current evaluation, an earlier DFT step's (faded and in
/// the long stale dashes), or guessed from experience while the graph is stale
/// (`crate::bond_experience`: solid, in [`EXPERIENCE_TINT`], at [`EXPERIENCE_BOND_ALPHA`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondLook {
    Fresh,
    Stale,
    Experience,
}

/// The "from experience" tint (user request 2026-11-13; display only): a bond experience calls
/// bonded is drawn this much of the way from its own colour to a soft mint — enough to tell it
/// from a fresh bond next to it, not so much that it reads as another kind (the 3D view's
/// background is dark in both themes, and the mint stays light on a light custom background).
pub const EXPERIENCE_TINT: [f32; 3] = [0.55, 0.92, 0.78];
pub const EXPERIENCE_TINT_MIX: f32 = 0.6;
/// Alpha of a bond from experience: between a stale bond's and a fresh one's — a guess, not this
/// geometry's density.
pub const EXPERIENCE_BOND_ALPHA: f32 = 0.7;
/// The text and background alpha of an angle or length label on a guessed bond (review of the
/// bonds from experience): between the stale labels' (110, 70) and the fresh ones' (235, 150), so
/// nothing guessed reads as this geometry's measurement; the text takes [`experience_tinted`].
pub const EXPERIENCE_LABEL_ALPHA: (u8, u8) = (180, 110);

/// A label's text colour on a guessed bond: its own colour moved [`EXPERIENCE_TINT_MIX`] of the
/// way to [`EXPERIENCE_TINT`], as the bond's lines are.
pub fn experience_tinted(rgb: [u8; 3]) -> [u8; 3] {
    std::array::from_fn(|i| {
        let c = rgb[i] as f32 / 255.0;
        ((c + (EXPERIENCE_TINT[i] - c) * EXPERIENCE_TINT_MIX) * 255.0).round().clamp(0.0, 255.0) as u8
    })
}

/// The bonds of `graph` between the nuclei at `pos` (the positions drawn now — the topology of the
/// last DFT graph between the CURRENT nuclei, spec §3.4), faded and broken into the long stale
/// dashes ([`dash_pieces`], [`STALE_DASH`]) when `stale`. Nothing when the graph's nucleus count is
/// not `pos.len()` (a graph with wrong indices is never drawn). The viewer draws through
/// [`bond_lines_drawn`]; this uniform form is the tests'.
#[cfg(test)]
pub fn bond_lines(pos: &[[f64; 3]], graph: &sim::BondGraph, stale: bool) -> Vec<LineVertex> {
    let look = if stale { BondLook::Stale } else { BondLook::Fresh };
    bond_lines_drawn(pos, graph, &vec![look; graph.bonds.len()])
}

/// [`bond_lines`] with each bond's own look (`looks[k]` for `graph.bonds[k]`; a missing one is
/// stale): a bond from experience keeps its kind's pattern and its strength, solid (no stale
/// dashes), tinted.
pub fn bond_lines_drawn(pos: &[[f64; 3]], graph: &sim::BondGraph, looks: &[BondLook]) -> Vec<LineVertex> {
    let mut out = Vec::new();
    if graph.n_nuclei != pos.len() {
        return out;
    }
    for (k, b) in graph.bonds.iter().enumerate() {
        let look = looks.get(k).copied().unwrap_or(BondLook::Stale);
        let stale = look == BondLook::Stale;
        let base = match look {
            BondLook::Fresh => BOND_ALPHA,
            BondLook::Stale => STALE_BOND_ALPHA,
            BondLook::Experience => EXPERIENCE_BOND_ALPHA,
        };
        let (Some(&a), Some(&c)) = (pos.get(b.i), pos.get(b.j)) else { continue };
        let (a, c) = (Vec3::from_array(a.map(|x| x as f32)), Vec3::from_array(c.map(|x| x as f32)));
        let axis = c - a;
        let len = axis.length();
        if len < 1e-6 {
            continue;
        }
        let mut rgb = if b.multi_centre { RING_BOND } else { BOND };
        if look == BondLook::Experience {
            rgb = std::array::from_fn(|i| rgb[i] + (EXPERIENCE_TINT[i] - rgb[i]) * EXPERIENCE_TINT_MIX);
        }
        let f = strength_fraction(b.rho_bcp);
        let colour = [rgb[0], rgb[1], rgb[2], base * (MIN_ALPHA + (1.0 - MIN_ALPHA) * f)];
        let radius = BOND_RADIUS * (MIN_WIDTH + (1.0 - MIN_WIDTH) * f);
        let (u, v) = plane_basis(axis / len);
        let pieces = dash_pieces(len, pattern(b.kind()), stale);
        for k in 0..BOND_STRANDS {
            let t = k as f32 * std::f32::consts::TAU / BOND_STRANDS as f32;
            let off = radius * (t.cos() * u + t.sin() * v);
            for &(s, e) in &pieces {
                if s == 0.0 && e == len {
                    // a solid bond: exactly its two nuclei's ends
                    lines::segment(&mut out, (a + off).to_array(), (c + off).to_array(), colour);
                } else {
                    lines::segment(&mut out, (a + off + axis * (s / len)).to_array(), (a + off + axis * (e / len)).to_array(), colour);
                }
            }
        }
    }
    out
}

/// The tint of a core-overlap pair (M7, spec §5.6): red, over the bond lines.
pub const CORE_OVERLAP: [f32; 4] = [1.0, 0.35, 0.3, 0.8];

/// Each pair (I, J, R) of `pairs` whose frozen cores overlap (`EvalStats::core_overlaps`) as a red strand bundle
/// between the nuclei at `pos` (the positions drawn now): the frozen-core approximation fails there, the step is
/// still taken, and the answer is never a training sample. Nothing for an index outside `pos`.
pub fn core_overlap_lines(pos: &[[f64; 3]], pairs: &[(usize, usize, f64)]) -> Vec<LineVertex> {
    let mut out = Vec::new();
    for &(i, j, _) in pairs {
        let (Some(&a), Some(&c)) = (pos.get(i), pos.get(j)) else { continue };
        let (a, c) = (Vec3::from_array(a.map(|x| x as f32)), Vec3::from_array(c.map(|x| x as f32)));
        let axis = c - a;
        let len = axis.length();
        if len < 1e-6 {
            continue;
        }
        let (u, v) = plane_basis(axis / len);
        for k in 0..BOND_STRANDS {
            let t = k as f32 * std::f32::consts::TAU / BOND_STRANDS as f32;
            let off = 1.6 * BOND_RADIUS * (t.cos() * u + t.sin() * v);
            lines::segment(&mut out, (a + off).to_array(), (c + off).to_array(), CORE_OVERLAP);
        }
    }
    out
}

/// How an angle between two drawn bonds is styled (plan review Q10): it follows the weaker of the
/// two bonds' QTAIM kinds — a multi-centre bond makes it `Ring`, a closed-shell contact `Contact`
/// (dashed, faint), an unresolved sign `Indeterminate` (dotted, never drawn as shared: rule 7),
/// both shared `Shared` (solid).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngleStyle {
    Shared,
    Contact,
    Indeterminate,
    Ring,
}

/// The angle a–`vertex`–b between two bonds meeting at `vertex` (a < b by nucleus index).
#[derive(Clone, Debug, PartialEq)]
pub struct AngleMark {
    pub vertex: usize,
    pub a: usize,
    pub b: usize,
    pub radians: f64,
    pub style: AngleStyle,
}

/// The arc's MINIMUM radius around the vertex (Bohr; display: [`angle_arc_radius`] raises it to clear
/// the vertex sphere) and the chords it is cut into per full arc.
pub const ANGLE_ARC_RADIUS: f32 = 0.45;
pub const ANGLE_ARC_SEGMENTS: usize = 16;
/// An arc is a bundle of this many thin lines on a tube of this radius (Bohr) round its curve, like a
/// bond's strand bundle but at half the bond's full radius and a fixed width: one pixel-wide line
/// reads as a hairline next to the spheres (display choice, not physics).
pub const ANGLE_STRANDS: usize = 4;
pub const ANGLE_TUBE_RADIUS: f32 = 0.5 * BOND_RADIUS;
/// The arc is drawn this many times clear of the vertex's drawn sphere radius, capped at this
/// fraction of the shorter of the two bonds (ruling: the 0.45 Bohr minimum sits inside an O sphere,
/// radius 0.70, and the depth test would hide it).
pub const ANGLE_ARC_CLEARANCE: f32 = 1.3;
pub const ANGLE_ARC_BOND_FRACTION: f32 = 0.6;
/// More bonds than this at a vertex: its angles are labelled only while it is selected or hovered
/// (CH₄'s C; water, NH₃ and every small molecule show theirs).
pub const ANGLE_DECLUTTER_BONDS: usize = 3;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// atan2(|u × w|, u·w): exact to rounding at every angle (acos loses digits near 0° and 180°).
fn angle_between(u: [f64; 3], w: [f64; 3]) -> f64 {
    let c = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(u[0] * w[0] + u[1] * w[1] + u[2] * w[2])
}

/// The style of the angle between two bonds (plan review Q10; M6's `sim::BondKind`): a multi-centre
/// bond → `Ring`; else a closed-shell contact → `Contact`; else an unresolved sign →
/// `Indeterminate` (never drawn as shared, rule 7); else `Shared`.
pub fn angle_style(a: &sim::Bond, b: &sim::Bond) -> AngleStyle {
    let kinds = [a.kind(), b.kind()];
    if a.multi_centre || b.multi_centre {
        AngleStyle::Ring
    } else if kinds.contains(&sim::BondKind::ClosedShell) {
        AngleStyle::Contact
    } else if kinds.contains(&sim::BondKind::Indeterminate) {
        AngleStyle::Indeterminate
    } else {
        AngleStyle::Shared
    }
}

/// Every angle between a pair of drawn bonds, from the nuclear positions `pos` (geometry only, no
/// physics input). Nothing when the graph's nucleus count is not `pos.len()` (as [`bond_lines`]).
pub fn bond_angles(pos: &[[f64; 3]], graph: &sim::BondGraph) -> Vec<AngleMark> {
    let mut out = Vec::new();
    if graph.n_nuclei != pos.len() {
        return out;
    }
    for v in 0..pos.len() {
        // a vertex with fewer than two bonds has no angle
        if graph.bonds.iter().filter(|b| b.i == v || b.j == v).count() < 2 {
            continue;
        }
        let at: Vec<(usize, &sim::Bond)> = graph
            .bonds
            .iter()
            .filter_map(|b| if b.i == v { Some((b.j, b)) } else if b.j == v { Some((b.i, b)) } else { None })
            .filter(|&(o, _)| o < pos.len())
            .collect();
        for x in 0..at.len() {
            for y in x + 1..at.len() {
                let ((a, ba), (b, bb)) = (at[x], at[y]);
                let (a, b, ba, bb) = if a < b { (a, b, ba, bb) } else { (b, a, bb, ba) };
                out.push(AngleMark { vertex: v, a, b, radians: angle_between(sub(pos[a], pos[v]), sub(pos[b], pos[v])), style: angle_style(ba, bb) });
            }
        }
    }
    out
}

/// The angles to draw: all of a vertex with at most [`ANGLE_DECLUTTER_BONDS`] bonds; those of a
/// vertex with more only while it is `selected` or `hovered`.
pub fn shown_angles<'a>(all: &'a [AngleMark], graph: &sim::BondGraph, selected: Option<usize>, hovered: Option<usize>) -> Vec<&'a AngleMark> {
    let bonds_at = |v: usize| graph.bonds.iter().filter(|b| b.i == v || b.j == v).count();
    all.iter().filter(|m| bonds_at(m.vertex) <= ANGLE_DECLUTTER_BONDS || selected == Some(m.vertex) || hovered == Some(m.vertex)).collect()
}

/// "104.4°".
pub fn angle_label(m: &AngleMark) -> String {
    format!("{:.1}°", m.radians.to_degrees())
}

/// The arc's radius (Bohr) at a vertex whose sphere is drawn with radius `vertex_radius`:
/// [`ANGLE_ARC_RADIUS`] at least, [`ANGLE_ARC_CLEARANCE`] × the drawn radius where that is more (so the
/// arc is outside its own sphere), at most [`ANGLE_ARC_BOND_FRACTION`] × the shorter of the two bonds
/// (the cap wins).
pub fn angle_arc_radius(pos: &[[f64; 3]], m: &AngleMark, vertex_radius: f32) -> f32 {
    let len = |i: usize| (Vec3::from_array(pos[i].map(|x| x as f32)) - Vec3::from_array(pos[m.vertex].map(|x| x as f32))).length();
    let r = ANGLE_ARC_RADIUS.max(ANGLE_ARC_CLEARANCE * vertex_radius);
    if m.a < pos.len() && m.b < pos.len() && m.vertex < pos.len() {
        r.min(ANGLE_ARC_BOND_FRACTION * len(m.a).min(len(m.b)))
    } else {
        r
    }
}

/// (vertex, u, n) of an angle's arc: the unit vector to `a` and the unit vector in the (u, w) plane
/// perpendicular to it (well defined below 180°; at 180° any perpendicular from [`plane_basis`]).
/// `None` for a degenerate angle (coincident nuclei, θ ≈ 0).
fn arc_frame(pos: &[[f64; 3]], m: &AngleMark) -> Option<(Vec3, Vec3, Vec3)> {
    let at = |i: usize| pos.get(i).map(|p| Vec3::from_array(p.map(|x| x as f32)));
    let (v, a, b) = (at(m.vertex)?, at(m.a)?, at(m.b)?);
    let u = (a - v).normalize_or_zero();
    let w = (b - v).normalize_or_zero();
    if u == Vec3::ZERO || w == Vec3::ZERO || m.radians < 1e-4 {
        return None;
    }
    let perp = w - u * u.dot(w);
    let n = if perp.length() > 1e-4 { perp.normalize() } else { plane_basis(u).0 };
    Some((v, u, n))
}

/// An arc round the vertex from the direction of `a` to that of `b`, of radius [`angle_arc_radius`]
/// (`vertex_radius`: the vertex sphere's drawn radius), faded and broken into the stale dashes
/// like the bonds when `stale` ([`dash_pieces`]).
///
/// The pattern is M6's bond pattern measured along the arc ([`pattern`]): `Shared` solid, `Contact`
/// dashed and faint (as M6's weakest contact), `Indeterminate` dotted (at one strand the dots would
/// be the weakest style; the bundle makes them read), a `Ring` angle dashed in the ring colour. Each
/// drawn piece is cut into chords at [`ANGLE_ARC_SEGMENTS`] per full arc, and each chord is
/// [`ANGLE_STRANDS`] lines on a tube of [`ANGLE_TUBE_RADIUS`] round the curve u·cos t + n·sin t.
#[cfg(test)]
pub fn angle_arc_lines(pos: &[[f64; 3]], m: &AngleMark, vertex_radius: f32, stale: bool) -> Vec<LineVertex> {
    angle_arc_lines_drawn(pos, m, vertex_radius, if stale { BondLook::Stale } else { BondLook::Fresh })
}

/// [`angle_arc_lines`] with the arc's look (the viewer's: stale when either bond is drawn stale,
/// else from experience when either bond is guessed): an arc between guessed bonds is solid in
/// its pattern, in [`EXPERIENCE_TINT`] at [`EXPERIENCE_BOND_ALPHA`], like the bonds it spans.
pub fn angle_arc_lines_drawn(pos: &[[f64; 3]], m: &AngleMark, vertex_radius: f32, look: BondLook) -> Vec<LineVertex> {
    let mut out = Vec::new();
    let Some((v, u, n)) = arc_frame(pos, m) else { return out };
    let radius = angle_arc_radius(pos, m, vertex_radius);
    let stale = look == BondLook::Stale;
    let base = match look {
        BondLook::Fresh => BOND_ALPHA,
        BondLook::Stale => STALE_BOND_ALPHA,
        BondLook::Experience => EXPERIENCE_BOND_ALPHA,
    };
    let (mut rgb, alpha, dash) = match m.style {
        AngleStyle::Shared => (BOND, base, pattern(sim::BondKind::Shared)),
        AngleStyle::Contact => (BOND, base * MIN_ALPHA, pattern(sim::BondKind::ClosedShell)),
        AngleStyle::Indeterminate => (BOND, base, pattern(sim::BondKind::Indeterminate)),
        AngleStyle::Ring => (RING_BOND, base, pattern(sim::BondKind::ClosedShell)),
    };
    if look == BondLook::Experience {
        rgb = std::array::from_fn(|i| rgb[i] + (EXPERIENCE_TINT[i] - rgb[i]) * EXPERIENCE_TINT_MIX);
    }
    let colour = [rgb[0], rgb[1], rgb[2], alpha];
    let theta = m.radians as f32;
    let len = radius * theta;
    // the tube's offsets lie in the plane perpendicular to the arc's tangent: the radial direction
    // and the normal of the bonds' plane
    let normal = u.cross(n);
    let point = |s: f32, k: usize| {
        let t = s / len * theta;
        let radial = u * t.cos() + n * t.sin();
        let phi = k as f32 * std::f32::consts::TAU / ANGLE_STRANDS as f32;
        v + radius * radial + ANGLE_TUBE_RADIUS * (phi.cos() * radial + phi.sin() * normal)
    };
    for (s0, s1) in dash_pieces(len, dash, stale) {
        let k = ((s1 - s0) / len * ANGLE_ARC_SEGMENTS as f32).ceil().max(1.0) as usize;
        for c in 0..k {
            let (a, b) = (s0 + (s1 - s0) * c as f32 / k as f32, s0 + (s1 - s0) * (c + 1) as f32 / k as f32);
            for strand in 0..ANGLE_STRANDS {
                lines::segment(&mut out, point(a, strand).to_array(), point(b, strand).to_array(), colour);
            }
        }
    }
    out
}

/// Where the label goes: the arc's midpoint, pushed out by a third of the radius.
pub fn angle_label_anchor(pos: &[[f64; 3]], m: &AngleMark, vertex_radius: f32) -> [f64; 3] {
    let Some((v, u, n)) = arc_frame(pos, m) else { return pos.get(m.vertex).copied().unwrap_or([0.0; 3]) };
    let h = 0.5 * m.radians as f32;
    (v + 1.33 * angle_arc_radius(pos, m, vertex_radius) * (u * h.cos() + n * h.sin())).to_array().map(|x| x as f64)
}

/// Grid spacing in Bohr for the panel's unit choice.
pub fn spacing(in_angstrom: bool) -> f32 {
    if in_angstrom { units::angstrom_to_bohr(1.0) as f32 } else { 1.0 }
}

/// In-plane orthonormal basis (u, v) of the plane with unit normal `n`: u horizontal where possible.
fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    let mut u = Vec3::Y.cross(n);
    if u.length_squared() < 1e-6 { u = Vec3::X.cross(n); }
    let u = u.normalize();
    (u, n.cross(u).normalize())
}

pub fn build(ui: &UiState, scene: &Scene, camera: &OrbitCamera, mode: ToolMode) -> Vec<LineVertex> {
    let mut out = Vec::new();
    // the cell is centred on the origin: samples at origin + i·h, origin = −(n−1)h/2 (qm::Grid::cube)
    let half = (scene.grid_n as f32 - 1.0) * scene.grid_h as f32 / 2.0;
    let s = spacing(ui.grid_in_angstrom);
    if ui.show_cell {
        lines::wire_box(&mut out, [-half; 3], [half; 3], CELL);
    }
    if let Some(w) = ui.walls() {
        let l = w.half_width as f32;
        lines::wire_box(&mut out, [-l; 3], [l; 3], WALLS);
    }
    if ui.show_floor {
        // cover the cell footprint in whole spacings
        let ext = (half / s).ceil() * s;
        lines::grid(&mut out, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], ext, s, MAJOR_EVERY, FLOOR_MINOR, FLOOR_MAJOR);
    }
    if ui.show_axes {
        lines::axes(&mut out, [0.0; 3], (2.0 * s).min(half));
    }
    if ui.show_place_grid && mode == ToolMode::Place {
        let (p, n) = camera.focus_plane();
        let (u, v) = plane_basis(n);
        lines::grid(&mut out, p.to_array(), u.to_array(), v.to_array(), PLACE_HALF_EXTENT, s, MAJOR_EVERY, PLACE_MINOR, PLACE_MAJOR);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ui(scene: &Scene) -> UiState {
        let mut ui = UiState::new("h2", scene);
        ui.show_cell = false;
        ui.show_floor = false;
        ui.show_axes = false;
        ui.show_place_grid = false;
        // the container walls are on by default in the app (they draw an inner box): off here,
        // explicitly — the tests below switch on what each one draws
        ui.walls_on = false;
        ui
    }

    #[test]
    fn everything_off_draws_nothing() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        assert!(build(&ui(&scene), &scene, &cam, ToolMode::Orbit).is_empty());
    }

    #[test]
    fn cell_box_matches_the_grid_extent() {
        let scene = Scene::preset("h2").unwrap(); // 49 × 0.25 → half = 6.0
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let mut u = ui(&scene);
        u.show_cell = true;
        let v = build(&u, &scene, &cam, ToolMode::Orbit);
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|p| p.pos.iter().all(|c| c.abs() == 6.0)));
    }

    #[test]
    fn container_walls_draw_an_inner_box() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let mut u = ui(&scene);
        u.walls_on = true;
        u.wall_half_width = 4.0;
        let v = build(&u, &scene, &cam, ToolMode::Orbit);
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|p| p.pos.iter().all(|c| c.abs() == 4.0) && p.colour == WALLS));
    }

    #[test]
    fn placement_grid_only_in_place_mode_and_lies_in_the_focus_plane() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::new(0.3, -0.2, 0.1), 10.0);
        let mut u = ui(&scene);
        u.show_place_grid = true;
        assert!(build(&u, &scene, &cam, ToolMode::Select).is_empty());
        let v = build(&u, &scene, &cam, ToolMode::Place);
        assert!(!v.is_empty());
        let (p, n) = cam.focus_plane();
        for q in &v {
            let d = (Vec3::from_array(q.pos) - p).dot(n);
            assert!(d.abs() < 1e-4, "vertex off the focus plane by {d}");
        }
    }

    fn graph(n: usize, bonds: &[(usize, usize, bool)]) -> sim::BondGraph {
        graph_with(n, bonds, RHO_STRONG, -1.0)
    }

    /// Bonds of density `rho` at the bond point and Laplacian `lap` (band 0.01).
    fn graph_with(n: usize, bonds: &[(usize, usize, bool)], rho: f64, lap: f64) -> sim::BondGraph {
        sim::BondGraph {
            bonds: bonds.iter().map(|&(i, j, m)| sim::Bond { i, j, rho_bcp: rho, enhancement: 1.3, s: 0.5, route: sim::Route::Saddle, multi_centre: m, laplacian_bcp: lap, laplacian_noise: 0.01 }).collect(),
            components: vec![(0..n).collect()],
            step: 0,
            sim_time: 0.0,
            n_nuclei: n,
        }
    }

    /// Spec §3.4 / plan Task 10 step 1: a bond is drawn between the CURRENT positions (a strand
    /// bundle of radius `BOND_RADIUS` around the segment between them), at alpha 0.3 when the graph
    /// is stale and full alpha when not; a graph whose nucleus count is not the scene's draws
    /// nothing. (A stale bond is also dashed: `a_stale_bond_is_dashed_and_a_fresh_one_keeps_its_pattern`.)
    #[test]
    fn bond_lines_follow_current_positions_with_the_stale_fade() {
        let g = graph(2, &[(0, 1, false)]);
        // positions that moved since the graph's evaluation: the lines follow them
        let pos = [[-1.0, 0.2, 0.0], [1.5, 0.2, 0.3]];
        let v = bond_lines(&pos, &g, false);
        assert_eq!(v.len(), 2 * BOND_STRANDS, "one bond, {BOND_STRANDS} strands");
        assert!(v.iter().all(|p| p.colour[3] == BOND_ALPHA));
        for pair in v.chunks(2) {
            for (p, end) in pair.iter().zip(pos) {
                let d = (Vec3::from_array(p.pos) - Vec3::from_array(end.map(|x| x as f32))).length();
                assert!((d - BOND_RADIUS).abs() < 1e-5, "a strand ends {d} from its nucleus");
            }
        }
        // stale: faded, and every piece on the strands' cylinder round the CURRENT segment
        let v = bond_lines(&pos, &g, true);
        assert!(!v.is_empty() && v.iter().all(|p| p.colour[3] == STALE_BOND_ALPHA));
        let (a, c) = (Vec3::from_array(pos[0].map(|x| x as f32)), Vec3::from_array(pos[1].map(|x| x as f32)));
        let axis = (c - a).normalize();
        for p in &v {
            let r = Vec3::from_array(p.pos) - a;
            let along = r.dot(axis);
            assert!(along > -1e-5 && along < (c - a).length() + 1e-5, "within the bond");
            assert!(((r - axis * along).length() - BOND_RADIUS).abs() < 1e-5, "on the strand cylinder");
        }
        assert_eq!(STALE_BOND_ALPHA, 0.3);
        // a nucleus count change: nothing (never a graph with wrong indices)
        assert!(bond_lines(&pos[..1], &g, false).is_empty());
        assert!(bond_lines(&[[0.0; 3]; 3], &g, false).is_empty());
        // a ring bond is told apart by its colour (its pattern is that of its kind)
        let ring = bond_lines(&pos, &graph(2, &[(0, 1, true)]), false);
        assert_eq!(ring.len(), 2 * BOND_STRANDS, "a shared ring bond is solid");
        assert_ne!(ring[0].colour, bond_lines(&pos, &g, false)[0].colour, "ring bonds are told apart");
    }

    /// The strength display (user request, M6): width and alpha follow log ρ_b between the floor
    /// and `RHO_STRONG`; shared is solid, closed-shell dashed, indeterminate dotted (more, shorter
    /// segments than the dashes).
    #[test]
    fn bond_lines_encode_strength_and_kind() {
        assert_eq!(strength_fraction(sim::bonds::RHO_FLOOR), 0.0);
        assert_eq!(strength_fraction(RHO_STRONG), 1.0);
        assert_eq!(strength_fraction(1e-4), 0.0);
        assert_eq!(strength_fraction(0.9), 1.0);
        assert_eq!(strength_fraction(f64::NAN), 0.0);
        assert!((strength_fraction((sim::bonds::RHO_FLOOR * RHO_STRONG).sqrt()) - 0.5).abs() < 1e-6, "log scale: the geometric mean is halfway");
        let pos = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let radius = |v: &[LineVertex]| (Vec3::from_array(v[0].pos) - Vec3::from_array([-1.5f32, 0.0, 0.0])).length();
        let strong = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.27, -1.0), false);
        let weak = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.02, -1.0), false);
        assert_eq!((strong.len(), weak.len()), (2 * BOND_STRANDS, 2 * BOND_STRANDS), "shared: solid");
        assert!(radius(&weak) < radius(&strong) && weak[0].colour[3] < strong[0].colour[3], "weaker is thinner and fainter");
        assert!((radius(&weak) - BOND_RADIUS * (MIN_WIDTH + (1.0 - MIN_WIDTH) * strength_fraction(0.02))).abs() < 1e-5);
        let dashed = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.02, 0.5), false);
        let dotted = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.02, 0.005), false);
        assert!(dashed.len() > 2 * BOND_STRANDS && dashed.len() % (2 * BOND_STRANDS) == 0, "closed-shell: dashed ({})", dashed.len());
        assert!(dotted.len() > dashed.len() && dotted.len() % (2 * BOND_STRANDS) == 0, "indeterminate: dotted ({} vs {})", dotted.len(), dashed.len());
        let seg = |v: &[LineVertex]| (Vec3::from_array(v[1].pos) - Vec3::from_array(v[0].pos)).length();
        assert!((seg(&dashed) - DASH).abs() < 1e-5 && (seg(&dotted) - DOT).abs() < 1e-5);
    }

    /// (start, end) along the bond of each piece of strand 0 (the first `pieces` segments).
    fn strand_pieces(v: &[LineVertex], a: [f64; 3], c: [f64; 3]) -> Vec<(f32, f32)> {
        let (a, c) = (Vec3::from_array(a.map(|x| x as f32)), Vec3::from_array(c.map(|x| x as f32)));
        let axis = (c - a).normalize();
        let n = v.len() / 2 / BOND_STRANDS;
        v.chunks(2).take(n).map(|s| ((Vec3::from_array(s[0].pos) - a).dot(axis), (Vec3::from_array(s[1].pos) - a).dot(axis))).collect()
    }

    /// User request 2026-11-13: a stale line is long dashes with a gap on its middle, mirror-symmetric,
    /// and each dash carries its kind's own pattern (two closed-shell dashes, four dots); a fresh line
    /// is its kind's pattern from the start, as before.
    #[test]
    fn stale_dash_pieces_are_long_dashes_round_a_middle_gap() {
        for len in [1.4f32, 2.06, 3.0, 5.2] {
            let p = dash_pieces(len, None, true);
            assert!(p.len() >= 2, "{len}: {p:?}");
            assert!(p.iter().all(|&(a, b)| 0.0 <= a && a < b && b <= len && b - a <= STALE_DASH + 1e-5), "{len}: {p:?}");
            assert!(p.iter().all(|&(a, b)| !(a < 0.5 * len && 0.5 * len < b)), "{len}: the middle is a gap: {p:?}");
            for w in p.windows(2) {
                assert!((w[1].0 - w[0].1 - STALE_GAP).abs() < 1e-5, "{len}: every gap is STALE_GAP: {p:?}");
            }
            for (x, y) in p.iter().zip(p.iter().rev()) {
                assert!((x.0 - (len - y.1)).abs() < 1e-4 && (x.1 - (len - y.0)).abs() < 1e-4, "{len}: symmetric about the middle: {p:?}");
            }
            assert_eq!(dash_pieces(len, None, false), vec![(0.0, len)], "fresh solid: one piece");
        }
        // a long line: its whole dashes are STALE_DASH long
        let p = dash_pieces(5.2, None, true);
        assert!(p.iter().filter(|&&(a, b)| a > 0.0 && b < 5.2).all(|&(a, b)| (b - a - STALE_DASH).abs() < 1e-5));
        // the gap is clearly longer than either kind's, the dash than either kind's period
        assert!(STALE_GAP > 1.5 * GAP.max(DOT_GAP) && STALE_DASH > DASH + GAP);
        // inside each whole stale dash, the kind's own pattern: two dashes or four dots
        let whole = |kind| {
            let all = dash_pieces(5.2, Some(kind), true);
            let outer = dash_pieces(5.2, None, true);
            let (a, b) = outer[outer.len() / 2];
            all.into_iter().filter(|&(s, e)| s >= a - 1e-5 && e <= b + 1e-5).collect::<Vec<_>>()
        };
        let dashes = whole((DASH, GAP));
        assert_eq!(dashes.len(), 2, "{dashes:?}");
        assert!(dashes.iter().all(|&(s, e)| (e - s - DASH).abs() < 1e-5));
        let dots = whole((DOT, DOT_GAP));
        assert_eq!(dots.len(), 4, "{dots:?}");
        // fresh: the kind's pattern from 0, unchanged
        let fresh = dash_pieces(2.0, Some((DASH, GAP)), false);
        assert_eq!(fresh[0], (0.0, DASH));
        assert!((fresh[1].0 - (DASH + GAP)).abs() < 1e-6);
        // shorter than the gap: still drawn
        assert!(!dash_pieces(0.1, None, true).is_empty());
        assert!(dash_pieces(0.0, None, true).is_empty());
    }

    /// User request 2026-11-13: a stale graph's bond is drawn as pieces with gaps (faded), a gap on
    /// the bond's middle; a fresh one keeps its old pattern — solid shared, closed-shell dashed from
    /// the first nucleus.
    #[test]
    fn a_stale_bond_is_dashed_and_a_fresh_one_keeps_its_pattern() {
        let pos = [[-1.0, 0.0, 0.0], [1.06, 0.0, 0.0]];
        let len = 2.06f32;
        let shared = graph(2, &[(0, 1, false)]);
        let fresh = bond_lines(&pos, &shared, false);
        assert_eq!(fresh.len(), 2 * BOND_STRANDS, "fresh shared: one solid segment per strand");
        assert_eq!(strand_pieces(&fresh, pos[0], pos[1]).len(), 1);
        let stale = bond_lines(&pos, &shared, true);
        assert!(stale.len() > 2 * BOND_STRANDS && stale.len() % (2 * BOND_STRANDS) == 0, "stale: several pieces per strand ({})", stale.len());
        let p = strand_pieces(&stale, pos[0], pos[1]);
        let drawn: f32 = p.iter().map(|&(a, b)| b - a).sum();
        assert!(drawn < len - STALE_GAP + 1e-4, "gaps left out: {drawn} of {len}");
        assert!(p.iter().all(|&(a, b)| !(a < 0.5 * len && 0.5 * len < b)), "the middle is a gap: {p:?}");
        assert!(stale.iter().all(|v| v.colour[3] == STALE_BOND_ALPHA), "and faded");
        // a closed-shell contact: fresh, its own dashes from the first nucleus; stale, other pieces
        let contact = graph_with(2, &[(0, 1, false)], 0.02, 0.5);
        let fresh = strand_pieces(&bond_lines(&pos, &contact, false), pos[0], pos[1]);
        let old = dash_pieces(len, Some((DASH, GAP)), false);
        assert_eq!(fresh.len(), old.len());
        assert!(fresh.iter().zip(&old).all(|(f, o)| (f.0 - o.0).abs() < 1e-5 && (f.1 - o.1).abs() < 1e-5), "{fresh:?} vs {old:?}");
        assert!(fresh[0].0.abs() < 1e-5 && (fresh[0].1 - DASH).abs() < 1e-5, "{fresh:?}");
        let stale = strand_pieces(&bond_lines(&pos, &contact, true), pos[0], pos[1]);
        let widest = |p: &[(f32, f32)]| p.windows(2).map(|w| w[1].0 - w[0].1).fold(0.0f32, f32::max);
        assert!((widest(&fresh) - GAP).abs() < 1e-4, "fresh: only the contact's own gaps {fresh:?}");
        assert!(widest(&stale) > STALE_GAP - 1e-4, "stale: the long stale gaps between groups {stale:?}");
        assert!(stale.iter().all(|&(a, b)| !(a < 0.5 * len && 0.5 * len < b)), "the middle is a gap: {stale:?}");
    }

    /// User request 2026-11-13 (bonds from experience): in one drawing, a bond from experience is
    /// solid — one piece per strand, no stale gap — in the mint tint at its own alpha, a closed-shell
    /// one from experience keeps its kind's dashes, and a stale one keeps the long stale dashes; the
    /// uniform form is the per-bond one with every bond alike.
    #[test]
    fn a_bond_from_experience_is_solid_and_tinted_beside_a_stale_one() {
        let pos = [[-1.0, 0.0, 0.0], [1.06, 0.0, 0.0], [3.5, 0.0, 0.0]];
        let g = graph(3, &[(0, 1, false), (1, 2, false)]);
        let v = bond_lines_drawn(&pos, &g, &[BondLook::Experience, BondLook::Stale]);
        let exp: Vec<&LineVertex> = v.iter().filter(|p| (p.colour[3] - EXPERIENCE_BOND_ALPHA).abs() < 1e-6).collect();
        let stale: Vec<&LineVertex> = v.iter().filter(|p| (p.colour[3] - STALE_BOND_ALPHA).abs() < 1e-6).collect();
        assert_eq!(exp.len() + stale.len(), v.len(), "every vertex is one or the other (full strength)");
        assert_eq!(exp.len(), 2 * BOND_STRANDS, "from experience: solid, one segment per strand");
        assert!(stale.len() > 2 * BOND_STRANDS, "stale: in pieces ({})", stale.len());
        let tinted: [f32; 3] = std::array::from_fn(|i| BOND[i] + (EXPERIENCE_TINT[i] - BOND[i]) * EXPERIENCE_TINT_MIX);
        assert!(exp.iter().all(|p| p.colour[..3] == tinted[..]), "the experience tint");
        assert!(stale.iter().all(|p| p.colour[..3] == BOND[..]), "stale: the bond's own colour");
        assert!(tinted.iter().zip(&BOND).any(|(a, b)| (a - b).abs() > 0.1), "a visible tint");
        // a closed-shell bond from experience: its own dashes, not the stale ones
        let contact = graph_with(2, &[(0, 1, false)], 0.02, 0.5);
        let p = strand_pieces(&bond_lines_drawn(&pos[..2], &contact, &[BondLook::Experience]), pos[0], pos[1]);
        let own = dash_pieces(2.06, Some((DASH, GAP)), false);
        assert_eq!(p.len(), own.len(), "{p:?} vs {own:?}");
        // uniform = per-bond with every look alike
        assert_eq!(bond_lines(&pos, &g, true), bond_lines_drawn(&pos, &g, &[BondLook::Stale; 2]));
        assert_eq!(bond_lines(&pos, &g, false), bond_lines_drawn(&pos, &g, &[BondLook::Fresh; 2]));
        assert!(bond_lines_drawn(&pos[..2], &g, &[BondLook::Fresh; 2]).is_empty(), "another nucleus count: nothing");
    }

    /// Review of the bonds from experience: an angle arc between guessed bonds is marked like them
    /// (solid, the mint tint, their alpha), and a label on one takes the same tint; a bond experience
    /// adds where the last graph had none is never fainter than any stale bond.
    #[test]
    fn guessed_arcs_and_labels_are_marked_and_an_added_bond_is_not_faint() {
        let pos = vec![[0.0; 3], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]];
        let m = AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(pos[1], pos[2]), style: AngleStyle::Shared };
        let fresh = angle_arc_lines_drawn(&pos, &m, 0.0, BondLook::Fresh);
        let guessed = angle_arc_lines_drawn(&pos, &m, 0.0, BondLook::Experience);
        assert_eq!(guessed.len(), fresh.len(), "solid, as a fresh arc");
        let tinted: [f32; 3] = std::array::from_fn(|i| BOND[i] + (EXPERIENCE_TINT[i] - BOND[i]) * EXPERIENCE_TINT_MIX);
        assert!(guessed.iter().all(|p| p.colour == [tinted[0], tinted[1], tinted[2], EXPERIENCE_BOND_ALPHA]), "the bonds' tint and alpha");
        assert_eq!(angle_arc_lines(&pos, &m, 0.0, true), angle_arc_lines_drawn(&pos, &m, 0.0, BondLook::Stale));
        // labels: the tint moves the text colour visibly, toward the mint
        let (length, angle) = ([205u8, 225, 245], [235u8, 225, 190]);
        for c in [length, angle] {
            let t = experience_tinted(c);
            assert!(t.iter().zip(&c).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap() >= 20, "{c:?} → {t:?}");
            assert!(t[1] >= c[1], "toward the mint's green");
        }
        let (a, bg) = EXPERIENCE_LABEL_ALPHA;
        assert!(110 < a && a < 235 && 70 < bg && bg < 150, "between the stale and the fresh labels");
        // an added bond (crate::bond_experience::plain_bond) against the strongest stale one
        let added = sim::BondGraph { bonds: vec![crate::bond_experience::plain_bond(0, 1)], components: vec![vec![0, 1]], step: 0, sim_time: 0.0, n_nuclei: 2 };
        let lines = bond_lines_drawn(&pos[..2], &added, &[BondLook::Experience]);
        let strongest_stale = STALE_BOND_ALPHA * (MIN_ALPHA + (1.0 - MIN_ALPHA) * 1.0);
        assert!(!lines.is_empty() && lines.iter().all(|p| p.colour[3] > strongest_stale), "{} vs {strongest_stale}", lines[0].colour[3]);
    }

    /// User request 2026-11-13: a stale arc is the stale dash along the arc (faded); a fresh one keeps
    /// its pattern.
    #[test]
    fn a_stale_arc_is_dashed_and_a_fresh_one_keeps_its_pattern() {
        let pos = vec![[0.0; 3], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]];
        let mark = |style| AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(pos[1], pos[2]), style };
        let fresh = angle_arc_lines(&pos, &mark(AngleStyle::Shared), 0.0, false);
        assert_eq!(fresh.len(), 2 * ANGLE_ARC_SEGMENTS * ANGLE_STRANDS, "fresh: solid, as before");
        let stale = angle_arc_lines(&pos, &mark(AngleStyle::Shared), 0.0, true);
        assert!(!stale.is_empty() && stale.len() < fresh.len(), "stale: gaps ({} of {})", stale.len(), fresh.len());
        assert!(stale.iter().all(|p| p.colour[3] == STALE_BOND_ALPHA));
        // the arc's middle direction is a gap: no chord end lies within the gap's half-width of it
        let m = mark(AngleStyle::Shared);
        let r = angle_arc_radius(&pos, &m, 0.0);
        let half = 0.5 * m.radians as f32;
        let mid = Vec3::new(half.cos(), half.sin(), 0.0);
        assert!(stale.iter().all(|p| Vec3::from_array(p.pos).normalize().angle_between(mid) * r > 0.5 * STALE_GAP - ANGLE_TUBE_RADIUS - 1e-3), "a gap on the middle");
    }

    /// M7 Task 9 (spec §5.6): an overlapping pair is a red strand bundle between the current positions; an
    /// index outside the scene draws nothing.
    #[test]
    fn core_overlap_pairs_are_red_segments_between_the_current_positions() {
        let pos = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let v = core_overlap_lines(&pos, &[(0, 1, 1.0)]);
        assert_eq!(v.len(), 2 * BOND_STRANDS);
        assert!(v.iter().all(|p| p.colour == CORE_OVERLAP));
        assert!(core_overlap_lines(&pos, &[(0, 7, 1.0)]).is_empty());
        assert!(core_overlap_lines(&pos, &[]).is_empty());
    }

    // ---- bond angles (M7 Task 13) ----

    /// A shared-shell bond (∇²ρ_b = −1 against a 0.1 band) unless `laplacian` says otherwise.
    fn abond(i: usize, j: usize, ring: bool, laplacian: f64) -> sim::Bond {
        sim::Bond { i, j, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: sim::Route::Saddle, multi_centre: ring, laplacian_bcp: laplacian, laplacian_noise: 0.1 }
    }

    fn agraph(n: usize, pairs: &[(usize, usize)], ring: bool) -> sim::BondGraph {
        sim::BondGraph {
            bonds: pairs.iter().map(|&(i, j)| abond(i, j, ring, -1.0)).collect(),
            components: vec![(0..n).collect()],
            step: 1,
            sim_time: 0.0,
            n_nuclei: n,
        }
    }

    /// Spec §8 row 13 first failing test: the water preset's angle from its nuclear positions, to 1e-9 rad.
    #[test]
    fn angles_of_water_preset_match_geometry() {
        let s = crate::scene::Scene::preset("h2o").unwrap();
        let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
        let marks = bond_angles(&pos, &agraph(3, &[(0, 1), (0, 2)], false));
        assert_eq!(marks.len(), 1);
        let m = &marks[0];
        assert_eq!((m.vertex, m.a, m.b), (0, 1, 2));
        // the preset: O at the origin, H at (±1.43, 1.11, 0) — the half-angle is atan2(1.43, 1.11)
        let expected = 2.0 * 1.43f64.atan2(1.11);
        assert!((m.radians - expected).abs() < 1e-9, "{} vs {expected}", m.radians);
        assert_eq!(angle_label(m), format!("{:.1}°", expected.to_degrees()));
        assert_eq!(shown_angles(&marks, &agraph(3, &[(0, 1), (0, 2)], false), None, None).len(), 1, "two bonds: always shown");
    }

    /// Spec §8 row 13: where four bonds meet (CH₄), only the selected (or hovered) atom's angles show.
    #[test]
    fn four_bonds_only_selected_atom_labelled() {
        let a = 2.06 / 3f64.sqrt();
        let pos = vec![[0.0; 3], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]];
        let g = agraph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)], false);
        let marks = bond_angles(&pos, &g);
        assert_eq!(marks.len(), 6);
        let tetra = (-1.0f64 / 3.0).acos();
        assert!(marks.iter().all(|m| (m.radians - tetra).abs() < 1e-9));
        assert!(shown_angles(&marks, &g, None, None).is_empty(), "four bonds, nothing selected: no labels");
        assert_eq!(shown_angles(&marks, &g, Some(0), None).len(), 6);
        assert_eq!(shown_angles(&marks, &g, None, Some(0)).len(), 6);
        assert!(shown_angles(&marks, &g, Some(1), None).is_empty(), "an H has one bond: no angle at it");
    }

    #[test]
    fn three_bonds_show_without_selection_and_ring_angles_are_ring_styled() {
        let pos = vec![[0.0; 3], [1.9, 0.0, -0.6], [-0.95, 1.65, -0.6], [-0.95, -1.65, -0.6]];
        let g = agraph(4, &[(0, 1), (0, 2), (0, 3)], false);
        let marks = bond_angles(&pos, &g);
        assert_eq!(shown_angles(&marks, &g, None, None).len(), 3);
        assert!(marks.iter().all(|m| m.style == AngleStyle::Shared));
        let ring = agraph(3, &[(0, 1), (1, 2), (0, 2)], true);
        let h3 = vec![[0.0, 0.0, 0.0], [1.65, 0.0, 0.0], [0.825, 1.429, 0.0]];
        let ring_marks = bond_angles(&h3, &ring);
        assert_eq!(ring_marks.len(), 3, "a triangle: one angle at each corner");
        assert!(ring_marks.iter().all(|m| m.style == AngleStyle::Ring));
        assert!(ring_marks.iter().all(|m| (m.radians - std::f64::consts::FRAC_PI_3).abs() < 1e-3), "the H₃⁺ ring is equilateral");
    }

    /// Plan review Q10: the angle takes the kinds of both bonds — a closed-shell contact makes it the
    /// contact's faint dash, an indeterminate one its own dotted style (never drawn as shared), a ring
    /// bond wins over both.
    #[test]
    fn angle_style_follows_the_bond_kinds() {
        let (shared, contact, indet) = (abond(0, 1, false, -1.0), abond(0, 2, false, 0.5), abond(0, 3, false, 0.05));
        assert_eq!(angle_style(&shared, &shared), AngleStyle::Shared);
        assert_eq!(angle_style(&shared, &contact), AngleStyle::Contact);
        assert_eq!(angle_style(&contact, &indet), AngleStyle::Contact);
        assert_eq!(angle_style(&indet, &shared), AngleStyle::Indeterminate);
        assert_eq!(angle_style(&abond(0, 4, false, f64::NAN), &shared), AngleStyle::Indeterminate, "a Laplacian that could not be evaluated");
        assert_eq!(angle_style(&abond(0, 1, true, 0.5), &shared), AngleStyle::Ring);
        // the patterns are M6's `pattern` along the arc (length 0.45·θ ≈ 0.83 Bohr here): solid draws all
        // ANGLE_ARC_SEGMENTS chords; the dash (0.22 on, 0.14 off) leaves out the gaps; the dots
        // (0.05 on, 0.12 off) leave out more
        let pos = vec![[0.0; 3], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]];
        let arc = |style| angle_arc_lines(&pos, &AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(pos[1], pos[2]), style }, 0.0, false).len();
        let (solid, dashed, dotted) = (arc(AngleStyle::Shared), arc(AngleStyle::Contact), arc(AngleStyle::Indeterminate));
        assert!(dotted < dashed && dashed < solid, "solid {solid}, dashed {dashed}, dotted {dotted} vertices");
        assert_eq!(solid, 2 * ANGLE_ARC_SEGMENTS * ANGLE_STRANDS, "a solid arc is ANGLE_ARC_SEGMENTS chords of ANGLE_STRANDS lines");
        // the colours and alphas: contact = the bond colour at the weakest contact's alpha, ring = the
        // ring colour and the contact's dash
        let mark = |style| AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(pos[1], pos[2]), style };
        let contact_arc = angle_arc_lines(&pos, &mark(AngleStyle::Contact), 0.0, false);
        assert!(contact_arc.iter().all(|p| p.colour == [BOND[0], BOND[1], BOND[2], BOND_ALPHA * MIN_ALPHA]));
        let ring_arc = angle_arc_lines(&pos, &mark(AngleStyle::Ring), 0.0, false);
        assert!(ring_arc.iter().all(|p| p.colour == [RING_BOND[0], RING_BOND[1], RING_BOND[2], BOND_ALPHA]));
        assert_eq!(ring_arc.len(), dashed, "a ring angle is dashed like a contact");
        let shared_arc = angle_arc_lines(&pos, &mark(AngleStyle::Shared), 0.0, false);
        assert!(shared_arc.iter().all(|p| p.colour == [BOND[0], BOND[1], BOND[2], BOND_ALPHA]));
    }

    /// The arc lies on the sphere of radius ANGLE_ARC_RADIUS round the vertex, in the bonds' plane,
    /// from the direction of a to that of b; it fades with a stale graph; the label sits outside it.
    #[test]
    fn the_arc_is_round_the_vertex_and_fades_when_stale() {
        let pos = vec![[0.3, -0.2, 0.1], [2.1, -0.2, 0.1], [-0.2, 1.5, 0.1]];
        let m = AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(sub(pos[1], pos[0]), sub(pos[2], pos[0])), style: AngleStyle::Shared };
        let v = Vec3::from_array(pos[0].map(|x| x as f32));
        let arc = angle_arc_lines(&pos, &m, 0.0, false);
        // every strand within the tube's radius of the arc, and the strands off the plane by no more
        // than the tube
        let tube = ANGLE_TUBE_RADIUS + 1e-4;
        assert!(arc.iter().all(|p| ((Vec3::from_array(p.pos) - v).length() - ANGLE_ARC_RADIUS).abs() <= tube));
        assert!(arc.iter().all(|p| (p.pos[2] - 0.1).abs() <= tube), "within a tube of the plane of the three nuclei");
        assert!(arc.iter().any(|p| (p.pos[2] - 0.1).abs() > 0.5 * ANGLE_TUBE_RADIUS), "and the strands do leave the plane: a bundle, not a line");
        let towards = |q: [f64; 3]| (Vec3::from_array(q.map(|x| x as f32)) - v).normalize();
        assert!((Vec3::from_array(arc[0].pos) - v).normalize().dot(towards(pos[1])) > 0.99, "starts along a");
        assert!((Vec3::from_array(arc.last().unwrap().pos) - v).normalize().dot(towards(pos[2])) > 0.99, "ends along b");
        assert!(arc.iter().all(|p| p.colour[3] == BOND_ALPHA));
        assert!(angle_arc_lines(&pos, &m, 0.0, true).iter().all(|p| p.colour[3] == STALE_BOND_ALPHA));
        let anchor = Vec3::from_array(angle_label_anchor(&pos, &m, 0.0).map(|x| x as f32));
        assert!(((anchor - v).length() - 1.33 * ANGLE_ARC_RADIUS).abs() < 1e-4);
        // a straight angle (180°) still has an arc, in some plane through the line
        let line = vec![[0.0; 3], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let straight = AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(line[1], line[2]), style: AngleStyle::Shared };
        assert!((straight.radians - std::f64::consts::PI).abs() < 1e-12);
        let arc = angle_arc_lines(&line, &straight, 0.0, false);
        assert!(!arc.is_empty() && arc.iter().all(|p| p.pos.iter().all(|c| c.is_finite())));
        // coincident nuclei: nothing, and no NaN
        let same = vec![[0.0; 3], [0.0; 3], [1.0, 0.0, 0.0]];
        let degenerate = AngleMark { vertex: 0, a: 1, b: 2, radians: 0.0, style: AngleStyle::Shared };
        assert!(angle_arc_lines(&same, &degenerate, 0.0, false).is_empty());
    }

    /// Ruling on the first screenshot: the 0.45 Bohr arc sat inside the O sphere (drawn radius 0.70) and
    /// the depth test hid it. At an O vertex the arc now clears its sphere; a small sphere keeps the
    /// 0.45 minimum; and a short bond caps it at 0.6 of the bond.
    #[test]
    fn an_o_vertex_arc_clears_its_sphere() {
        let s = crate::scene::Scene::preset("h2o").unwrap();
        let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
        let m = &bond_angles(&pos, &agraph(3, &[(0, 1), (0, 2)], false))[0];
        let r_o = render::elements::radius(8);
        assert_eq!(r_o, 0.70);
        let r = angle_arc_radius(&pos, m, r_o);
        assert!(r >= ANGLE_ARC_CLEARANCE * r_o - 1e-6 && r > r_o, "arc radius {r} vs sphere {r_o}");
        let v = Vec3::from_array(pos[0].map(|x| x as f32));
        let arc = angle_arc_lines(&pos, m, r_o, false);
        assert!(arc.iter().all(|p| (Vec3::from_array(p.pos) - v).length() > r_o), "every strand outside the sphere");
        let anchor = Vec3::from_array(angle_label_anchor(&pos, m, r_o).map(|x| x as f32));
        assert!(((anchor - v).length() - 1.33 * r).abs() < 1e-4);
        let r_h = render::elements::radius(1);
        assert!((angle_arc_radius(&pos, m, r_h) - (ANGLE_ARC_CLEARANCE * r_h)).abs() < 1e-6, "H: 1.3 × 0.35 = 0.455, just above the minimum");
        assert_eq!(angle_arc_radius(&pos, m, 0.2), ANGLE_ARC_RADIUS, "a small sphere: the minimum");
        // bonds of 0.5 Bohr: the cap (0.6 × 0.5) wins over both the minimum and the clearance
        let short = vec![[0.0; 3], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]];
        let sm = AngleMark { vertex: 0, a: 1, b: 2, radians: std::f64::consts::FRAC_PI_2, style: AngleStyle::Shared };
        assert!((angle_arc_radius(&short, &sm, r_o) - 0.3).abs() < 1e-6);
    }

    /// atan2(|u × w|, u·w) is exact near 0° and 180°, where acos is not.
    #[test]
    fn angle_between_is_well_conditioned_at_both_ends() {
        let tiny = 1e-9;
        let near_zero = angle_between([1.0, 0.0, 0.0], [1.0, tiny, 0.0]);
        assert!((near_zero - tiny).abs() < 1e-18, "{near_zero}");
        let near_pi = angle_between([1.0, 0.0, 0.0], [-1.0, tiny, 0.0]);
        assert!((near_pi - (std::f64::consts::PI - tiny)).abs() < 1e-15, "{near_pi}");
    }

    /// Nothing when the graph belongs to another nucleus count (as `bond_lines`).
    #[test]
    fn a_graph_of_another_scene_draws_no_angles() {
        assert!(bond_angles(&[[0.0; 3], [1.0, 0.0, 0.0]], &agraph(3, &[(0, 1), (0, 2)], false)).is_empty());
    }

    #[test]
    fn angstrom_spacing_changes_the_floor_line_count() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let mut u = ui(&scene);
        u.show_floor = true;
        let bohr = build(&u, &scene, &cam, ToolMode::Orbit).len();
        u.grid_in_angstrom = true;
        let ang = build(&u, &scene, &cam, ToolMode::Orbit).len();
        assert!(ang < bohr, "1 Å = 1.89 Bohr → fewer lines ({ang} vs {bohr})");
    }
}
