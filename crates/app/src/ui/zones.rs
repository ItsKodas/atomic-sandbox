//! The 3D view's overlays laid out in stacking zones (spec
//! `docs/superpowers/specs/2026-11-22-view-overlay-zones-design.md`, approach 1, user 2026-11-22).
//!
//! Every overlay over the view reports the size it took last frame in each of its forms (full
//! first, then its compact forms), and [`layout`] gives each a rect and a form, so that nothing
//! overlaps and everything stays inside the view. The pass is pure — rects in, rects out — and is
//! unit-tested without egui; each overlay then draws at its rect with `fixed_pos` and the pivot of
//! its zone ([`Slot::pivot`]).
//!
//! **The zones** (each a stack in a fixed order, [`Slot`]'s declaration order):
//! - top-left, downward from the corner: the status chip, the draft banner;
//! - top centre: the training ghost's tag ("training is solving · …", drawn by the viewer), in the
//!   top band when it is clear there, else under whatever is in the way;
//! - the right edge, downward from the top-right corner: the capture buttons (with their notes),
//!   the segment card, the offer cards, then the atom card, which scrolls when the column runs
//!   short (its height may shrink to [`Item::min_height`]);
//! - bottom-left, upward: the view switch, the Camera sliders, the Dots sliders, the orbital
//!   legend, the knowledge legend;
//! - bottom-centre, upward: the sampling card, the relax card, the density label, the toast;
//! - bottom-right, upward: the bonds / orbitals pair, the stale bonds' line, the Orbitals card.
//!
//! **The pass.** The edge stacks are laid first (top-left, bottom-right, bottom-left, then the right
//! column, which stops above the bottom-right stack), then the top-centre tag, then bottom-centre in the band between
//! bottom-left and bottom-right: centred on the view as far as its row allows. When anything
//! overlaps (closer than [`GAP`]) or leaves the view:
//! 1. while bottom-centre meets bottom-left or bottom-right, the widest of the items involved
//!    whose next compact form is narrower takes it (a one-row card, two slider rows);
//! 2. then bottom-centre leaves the bottom row: each of its items goes to the lowest place above
//!    its stack's last that clears everything (the second row, above bottom-left and bottom-right);
//! 3. then, for any overlap left, the item involved with the mildest next form takes it (a fold
//!    of the help lines, a one-row card), the tallest first (when none of them has one left, any
//!    item: what pushed them together, like a tall chip over a floated card); the last resorts — the chip's help
//!    lines behind their ⓘ alone, the sliders in a popover — come after every mild form;
//! 4. once nothing overlaps, a bottom-centre card compacted in step 1 that moved up anyway gets its
//!    full form back where the second row has room for it.
//!
//! The decision is a function of the sizes alone, so a zone's items move only when one appears,
//! disappears or changes form (spec anti-goal: nothing jumps frame to frame).
use egui::{pos2, Align2, Pos2, Rect, Vec2};
use std::collections::HashMap;

/// The gap between two overlays, in a stack and across zones (points).
pub const GAP: f32 = 6.0;
/// The top zones' inset from the view's edges (the status chip's and the capture buttons' as
/// before the zones).
pub const INSET_TOP: f32 = 12.0;
/// The bottom zones' inset from the view's edges (the view switch's and the bonds pair's).
pub const INSET_BOTTOM: f32 = 14.0;
/// The smallest view the zones promise to fit, with every compact form and bottom-centre's second
/// row (spec "The minimum view"); below it only the absence of overlap is aimed at. (The tests'
/// smallest size.)
#[cfg_attr(not(test), allow(dead_code))]
pub const MIN_VIEW: Vec2 = Vec2::new(560.0, 360.0);

/// An overlay of the view. The declaration order is each zone's stacking order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Slot {
    // top-left, downward
    Chip,
    Banner,
    // top centre
    CentreTag,
    // the right edge, downward
    Capture,
    Segment,
    Offer,
    TeachPrompt,
    AtomCard,
    // bottom-left, upward
    ViewSwitch,
    CameraSliders,
    DotsSliders,
    OrbitalLegend,
    KnowledgeLegend,
    // bottom-centre, upward
    Sampling,
    Relax,
    DensityLabel,
    Toast,
    // bottom-right, upward
    BondsPair,
    StaleLine,
    OrbitalsCard,
}

/// The six zones of the view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Zone {
    TopLeft,
    TopCentre,
    Right,
    BottomLeft,
    BottomCentre,
    BottomRight,
}

impl Slot {
    /// Every overlay, in order (the tests' "everything at once").
    #[cfg_attr(not(test), allow(dead_code))]
    pub const ALL: [Slot; 20] = [
        Slot::Chip,
        Slot::Banner,
        Slot::CentreTag,
        Slot::Capture,
        Slot::Segment,
        Slot::Offer,
        Slot::TeachPrompt,
        Slot::AtomCard,
        Slot::ViewSwitch,
        Slot::CameraSliders,
        Slot::DotsSliders,
        Slot::OrbitalLegend,
        Slot::KnowledgeLegend,
        Slot::Sampling,
        Slot::Relax,
        Slot::DensityLabel,
        Slot::Toast,
        Slot::BondsPair,
        Slot::StaleLine,
        Slot::OrbitalsCard,
    ];

    pub fn zone(self) -> Zone {
        use Slot::*;
        match self {
            Chip | Banner => Zone::TopLeft,
            CentreTag => Zone::TopCentre,
            Capture | Segment | Offer | TeachPrompt | AtomCard => Zone::Right,
            ViewSwitch | CameraSliders | DotsSliders | OrbitalLegend | KnowledgeLegend => Zone::BottomLeft,
            Sampling | Relax | DensityLabel | Toast => Zone::BottomCentre,
            BondsPair | StaleLine | OrbitalsCard => Zone::BottomRight,
        }
    }

    /// The corner of its rect an overlay is pinned by (its `Area`'s pivot): the corner its stack
    /// grows from, so a size that changed since the measurement grows away from the view's edge.
    pub fn pivot(self) -> Align2 {
        match self.zone() {
            Zone::TopLeft => Align2::LEFT_TOP,
            Zone::TopCentre => Align2::CENTER_TOP,
            Zone::Right => Align2::RIGHT_TOP,
            Zone::BottomLeft => Align2::LEFT_BOTTOM,
            Zone::BottomCentre => Align2::CENTER_BOTTOM,
            Zone::BottomRight => Align2::RIGHT_BOTTOM,
        }
    }

    /// How hard stepping to `form` is: 1 for a compact form that keeps everything on screen (a
    /// fold with its ⓘ, a one-row card, two slider rows), 2 for the last resorts (the chip's help
    /// lines behind one ⓘ, the sliders in a popover).
    fn severity(self, form: usize) -> u8 {
        match (self, form) {
            (Slot::Chip | Slot::DotsSliders | Slot::CameraSliders, 2) => 2,
            _ => 1,
        }
    }

    fn bottom(self) -> bool {
        matches!(self.zone(), Zone::BottomLeft | Zone::BottomCentre | Zone::BottomRight)
    }
}

/// One overlay's request: its forms' sizes.
#[derive(Clone, Debug)]
pub struct Item {
    pub slot: Slot,
    /// Each form's size (points), the full form first. A zero size takes no room in the zone (the
    /// sliders in their popover, or an overlay not measured yet).
    pub forms: Vec<Vec2>,
    /// The least height the item may be shrunk to (it scrolls): the atom card, last in its column.
    pub min_height: Option<f32>,
}

impl Item {
    pub fn new(slot: Slot, forms: Vec<Vec2>) -> Self {
        assert!(!forms.is_empty(), "an item has a full form");
        Item { slot, forms, min_height: None }
    }

    pub fn shrinkable(mut self, min_height: f32) -> Self {
        self.min_height = Some(min_height);
        self
    }
}

/// Where an overlay goes and in which form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub rect: Rect,
    pub form: usize,
}

/// The pass's answer for one frame.
#[derive(Clone, Debug)]
pub struct ViewLayout {
    pub view: Rect,
    placed: Vec<(Slot, Placed)>,
    /// Nothing was left to compact and something still overlaps or leaves the view (a view below
    /// [`MIN_VIEW`], or more at once than it holds). Read by the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub crowded: bool,
    /// The top-left stack's width at most: up to the right column's items beside it (next frame's
    /// status chip and banner wrap inside it).
    pub top_left_room: f32,
}

impl Default for ViewLayout {
    fn default() -> Self {
        ViewLayout { view: Rect::NOTHING, placed: Vec::new(), crowded: false, top_left_room: 0.0 }
    }
}

impl ViewLayout {
    pub fn get(&self, slot: Slot) -> Option<Placed> {
        self.placed.iter().find(|(s, _)| *s == slot).map(|(_, p)| *p)
    }

    /// The form `slot` is drawn in (the full form, 0, when it was not laid out).
    pub fn form(&self, slot: Slot) -> usize {
        self.get(slot).map_or(0, |p| p.form)
    }

    /// The pivot and the point an overlay is pinned at.
    pub fn pin(&self, slot: Slot) -> Option<(Align2, Pos2)> {
        self.get(slot).map(|p| (slot.pivot(), slot.pivot().pos_in_rect(&p.rect)))
    }

    /// Every overlay that takes room, with its rect.
    pub fn rects(&self) -> impl Iterator<Item = (Slot, Rect)> + '_ {
        self.placed.iter().filter(|(_, p)| takes_room(p.rect)).map(|(s, p)| (*s, p.rect))
    }

    /// The top of the bottom-right stack (the view's bottom inset without one): the capture
    /// settings' popover stops above it.
    pub fn bottom_right_top(&self) -> f32 {
        self.rects().filter(|(s, _)| s.zone() == Zone::BottomRight).map(|(_, r)| r.top()).fold(self.view.bottom() - INSET_BOTTOM, f32::min)
    }
}

/// The sizes each overlay measured, per form (kept in `UiState`; last frame's are this frame's
/// request).
#[derive(Default)]
pub struct Zones {
    sizes: HashMap<(Slot, usize), Vec2>,
    /// The layout drawn this frame (the viewer's ghost tag reads its slot).
    pub layout: ViewLayout,
    /// The ghost tag's size, set by the viewer as it draws it (after the chrome: a frame late).
    pub tag: std::cell::Cell<Option<Vec2>>,
}

impl Zones {
    /// Records the size `slot` took in `form`.
    pub fn measured(&mut self, slot: Slot, form: usize, size: Vec2) {
        self.sizes.insert((slot, form), size);
    }

    /// The sizes of `slot`'s forms: as measured, else the guesses given (the full form's guess is
    /// usually zero: it is measured on its first frame).
    pub fn forms(&self, slot: Slot, guesses: &[Vec2]) -> Vec<Vec2> {
        guesses.iter().enumerate().map(|(k, g)| self.sizes.get(&(slot, k)).copied().unwrap_or(*g)).collect()
    }

    /// The last measured size of `slot` in `form`.
    pub fn size(&self, slot: Slot, form: usize) -> Option<Vec2> {
        self.sizes.get(&(slot, form)).copied()
    }
}

fn takes_room(r: Rect) -> bool {
    r.width() > 0.5 && r.height() > 0.5
}

/// Two overlays closer than [`GAP`] (half a point of slack for rounding).
fn near(a: Rect, b: Rect) -> bool {
    let e = GAP / 2.0 - 0.5;
    a.expand(e).intersects(b.expand(e))
}

/// Lays the view's overlays out (see the module doc).
pub fn layout(view: Rect, items: &[Item]) -> ViewLayout {
    let mut forms = vec![0usize; items.len()];
    let mut float = false;
    let mut crowded = false;
    let mut rects;
    // the bottom-centre items compacted to fit the bottom row, in order (step 4 below)
    let mut row_compacted: Vec<usize> = Vec::new();
    // every step takes one form or the float, so this ends; the bound is a guard
    let mut steps = 0;
    loop {
        rects = place(view, items, &forms, float);
        let clashes = clashes_of(view, &rects);
        if clashes.is_empty() {
            // 4. bottom-centre left the row anyway: its cards compacted for the row get their
            // full form back where the second row has room for it
            if float {
                for &i in row_compacted.iter().rev() {
                    forms[i] -= 1;
                    let r = place(view, items, &forms, float);
                    if clashes_of(view, &r).is_empty() {
                        rects = r;
                    } else {
                        forms[i] += 1;
                    }
                }
            }
            break;
        }
        steps += 1;
        if steps > 200 {
            crowded = true;
            break;
        }
        let mut involved: Vec<usize> = clashes.iter().flat_map(|&(a, b)| std::iter::once(a).chain(b)).collect();
        involved.sort_unstable();
        involved.dedup();
        let next = |i: usize, forms: &[usize]| (forms[i] + 1 < items[i].forms.len()).then_some(forms[i] + 1);
        let size = |i: usize, forms: &[usize]| items[i].forms[forms[i]];
        if !float {
            let row = clashes.iter().any(|&(a, b)| {
                b.is_some_and(|b| {
                    let (za, zb) = (items[a].slot.zone(), items[b].slot.zone());
                    (za == Zone::BottomCentre) != (zb == Zone::BottomCentre) && items[a].slot.bottom() && items[b].slot.bottom()
                })
            });
            if row {
                // 1. the widest item involved whose next mild form is narrower
                let pick = involved
                    .iter()
                    .copied()
                    .filter(|&i| items[i].slot.bottom())
                    .filter(|&i| next(i, &forms).is_some_and(|f| items[i].slot.severity(f) == 1 && items[i].forms[f].x < size(i, &forms).x - 0.5))
                    .max_by(|&a, &b| size(a, &forms).x.total_cmp(&size(b, &forms).x));
                if let Some(i) = pick {
                    forms[i] += 1;
                    if items[i].slot.zone() == Zone::BottomCentre {
                        row_compacted.push(i);
                    }
                } else {
                    // 2. bottom-centre leaves the row
                    float = true;
                }
                continue;
            }
        }
        // 3. the mildest next form among the items involved, the tallest first; when none of them
        // has one, among every item (what pushed them together: a tall chip over a floated card)
        let mildest = |among: &mut dyn Iterator<Item = usize>| {
            among
                .filter_map(|i| next(i, &forms).map(|f| (i, items[i].slot.severity(f))))
                .min_by(|a, b| a.1.cmp(&b.1).then(size(b.0, &forms).y.total_cmp(&size(a.0, &forms).y)))
        };
        let pick = mildest(&mut involved.iter().copied()).or_else(|| if float { mildest(&mut (0..items.len())) } else { None });
        if let Some((i, _)) = pick {
            forms[i] += 1;
        } else if !float {
            float = true;
        } else {
            crowded = true;
            break;
        }
    }
    let placed: Vec<(Slot, Placed)> = items.iter().zip(&rects).zip(&forms).map(|((it, r), &form)| (it.slot, Placed { rect: *r, form })).collect();
    // the top-left stack's room: up to the right column's items beside it
    let tl_bottom = placed.iter().filter(|(s, p)| s.zone() == Zone::TopLeft && takes_room(p.rect)).map(|(_, p)| p.rect.bottom()).fold(view.top() + INSET_TOP, f32::max);
    let right_left = placed
        .iter()
        .filter(|(s, p)| s.zone() == Zone::Right && takes_room(p.rect) && (p.rect.top() < tl_bottom + GAP || *s == Slot::Capture))
        .map(|(_, p)| p.rect.left())
        .fold(view.right() - INSET_TOP, f32::min);
    let top_left_room = (right_left - GAP - (view.left() + INSET_TOP)).max(0.0);
    ViewLayout { view, placed, crowded, top_left_room }
}

/// The pairs of overlays closer than the gap, and (`None` as the second) those leaving the view.
fn clashes_of(view: Rect, rects: &[Rect]) -> Vec<(usize, Option<usize>)> {
    let mut v = Vec::new();
    for (i, a) in rects.iter().enumerate().filter(|(_, r)| takes_room(**r)) {
        if !view.expand(0.5).contains_rect(*a) {
            v.push((i, None));
        }
        for (j, b) in rects.iter().enumerate().skip(i + 1).filter(|(_, r)| takes_room(**r)) {
            if near(*a, *b) {
                v.push((i, Some(j)));
            }
        }
    }
    v
}

/// Every item's rect for these forms (a zero-size rect, at its stack's place, for an item that
/// takes no room).
fn place(view: Rect, items: &[Item], forms: &[usize], float: bool) -> Vec<Rect> {
    let mut out = vec![Rect::NOTHING; items.len()];
    let mut done = vec![false; items.len()];
    let in_zone = |z: Zone| {
        let mut v: Vec<usize> = (0..items.len()).filter(|&i| items[i].slot.zone() == z).collect();
        v.sort_by_key(|&i| items[i].slot);
        v
    };
    let size = |i: usize| items[i].forms[forms[i]];
    let room = |s: Vec2| s.x > 0.5 && s.y > 0.5;
    // top-left, downward from the corner
    let mut y = view.top() + INSET_TOP;
    for i in in_zone(Zone::TopLeft) {
        let s = if room(size(i)) { size(i) } else { Vec2::ZERO };
        out[i] = Rect::from_min_size(pos2(view.left() + INSET_TOP, y), s);
        if room(s) {
            y = out[i].bottom() + GAP;
        }
        done[i] = true;
    }
    // the bottom corners, upward
    // (bottom-left first: a bottom-right item that would meet it on a narrow view goes up past it)
    for (zone, left) in [(Zone::BottomLeft, true), (Zone::BottomRight, false)] {
        let mut y = view.bottom() - INSET_BOTTOM;
        for i in in_zone(zone) {
            let s = if room(size(i)) { size(i) } else { Vec2::ZERO };
            let x = if left { view.left() + INSET_BOTTOM } else { view.right() - INSET_BOTTOM - s.x };
            out[i] = Rect::from_min_size(pos2(x, y - s.y), s);
            if !left && room(s) {
                // up past whatever of bottom-left it meets (each step goes higher: this ends)
                loop {
                    let me = out[i];
                    let hit = out.iter().zip(&done).filter(|(r, d)| **d && takes_room(**r)).map(|(r, _)| *r).find(|o| near(*o, me));
                    match hit {
                        Some(o) if o.top() - GAP - s.y >= view.top() => out[i] = Rect::from_min_size(pos2(x, o.top() - GAP - s.y), s),
                        _ => break,
                    }
                }
            }
            if room(s) {
                y = out[i].top() - GAP;
            }
            done[i] = true;
        }
    }
    // the right edge, downward; the last item may shrink to stop above the bottom-right stack
    let mut y = view.top() + INSET_TOP;
    let x1 = view.right() - INSET_TOP;
    // the top-left stack, which a right-edge card too wide to sit beside goes under
    let top_left: Vec<Rect> = in_zone(Zone::TopLeft).into_iter().map(|i| out[i]).filter(|r| takes_room(*r)).collect();
    for i in in_zone(Zone::Right) {
        let mut s = if room(size(i)) { size(i) } else { Vec2::ZERO };
        // (the capture buttons keep the corner: the chip's room always stops at them)
        if room(s) && items[i].slot != Slot::Capture {
            if let Some(b) = top_left.iter().filter(|r| near(**r, Rect::from_min_size(pos2(x1 - s.x, y), s))).map(|r| r.bottom()).reduce(f32::max) {
                y = b + GAP;
            }
        }
        if let (Some(min_h), true) = (items[i].min_height, room(s)) {
            let floor = out
                .iter()
                .zip(&done)
                .filter(|(r, d)| **d && takes_room(**r) && r.bottom() > y && r.left() < x1 && r.right() > x1 - s.x)
                .map(|(r, _)| r.top() - GAP)
                .fold(view.bottom() - INSET_TOP, f32::min);
            s.y = s.y.min(floor - y).max(min_h);
        }
        out[i] = Rect::from_min_size(pos2(x1 - s.x, y), s);
        if room(s) {
            y = out[i].bottom() + GAP;
        }
        done[i] = true;
    }
    // the top-centre tag: the top band when clear there, else under what is in the way (before
    // bottom-centre, which floats round it)
    for i in in_zone(Zone::TopCentre) {
        let s = size(i);
        let obstacles: Vec<Rect> = out.iter().zip(&done).filter(|(r, d)| **d && takes_room(**r)).map(|(r, _)| *r).collect();
        out[i] = if room(s) { fit_down(view, s, &obstacles) } else { Rect::from_min_size(pos2(view.center().x, view.top() + INSET_TOP), Vec2::ZERO) };
        done[i] = true;
    }
    // bottom-centre, upward: in the bottom row, or (float) the lowest place that clears the rest
    let mut start = view.bottom() - INSET_BOTTOM;
    for i in in_zone(Zone::BottomCentre) {
        let s = size(i);
        if !room(s) {
            out[i] = Rect::from_min_size(pos2(view.center().x, start), Vec2::ZERO);
            done[i] = true;
            continue;
        }
        let obstacles: Vec<Rect> = out.iter().zip(&done).filter(|(r, d)| **d && takes_room(**r)).map(|(r, _)| *r).collect();
        out[i] = fit_up(view, s, start, &obstacles, float);
        start = out[i].top() - GAP;
        done[i] = true;
    }
    out
}

/// The left edge for a rect of `size` spanning `top..bottom`: centred on the view when that is
/// clear, else as near the centre as the gap between the obstacles beside it allows; `None` when
/// the gap is too narrow.
fn row_x(view: Rect, size: Vec2, top: f32, bottom: f32, obstacles: &[Rect]) -> Option<f32> {
    let (mut lo, mut hi) = (view.left() + INSET_TOP, view.right() - INSET_TOP);
    let cx = view.center().x;
    for o in obstacles.iter().filter(|o| o.top() < bottom + GAP - 0.5 && o.bottom() > top - GAP + 0.5) {
        if o.center().x < cx {
            lo = lo.max(o.right() + GAP);
        } else {
            hi = hi.min(o.left() - GAP);
        }
    }
    (hi - lo >= size.x - 0.01).then(|| (cx - size.x / 2.0).clamp(lo, (hi - size.x).max(lo)))
}

/// A rect of `size` with its bottom at `start` (centred as its row allows), or, searching, at the
/// lowest place at or above `start` that clears every obstacle.
fn fit_up(view: Rect, size: Vec2, start: f32, obstacles: &[Rect], search: bool) -> Rect {
    let mut ys = vec![start];
    if search {
        ys.extend(obstacles.iter().map(|o| o.top() - GAP).filter(|&y| y < start));
        ys.sort_by(|a, b| b.total_cmp(a));
    }
    for &y in &ys {
        if search && y - size.y < view.top() + INSET_TOP - 0.01 {
            break;
        }
        if let Some(x) = row_x(view, size, y - size.y, y, obstacles) {
            return Rect::from_min_size(pos2(x, y - size.y), size);
        }
    }
    Rect::from_min_size(pos2(view.center().x - size.x / 2.0, start - size.y), size)
}

/// A rect of `size` at the top band's centre when clear there, else at the highest place under
/// it that clears every obstacle.
fn fit_down(view: Rect, size: Vec2, obstacles: &[Rect]) -> Rect {
    let start = view.top() + INSET_TOP;
    let mut ys = vec![start];
    ys.extend(obstacles.iter().map(|o| o.bottom() + GAP).filter(|&y| y > start));
    ys.sort_by(|a, b| a.total_cmp(b));
    for &y in &ys {
        if y + size.y > view.bottom() - INSET_BOTTOM + 0.01 {
            break;
        }
        if let Some(x) = row_x(view, size, y, y + size.y, obstacles) {
            return Rect::from_min_size(pos2(x, y), size);
        }
    }
    Rect::from_min_size(pos2(view.center().x - size.x / 2.0, start), size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::vec2;

    /// The overlays' sizes as drawn (points), following each overlay's own width rule at a view
    /// `v` and a top-left room `room` (the chip and the banner wrap inside it): full forms first,
    /// then the compact ones. Measured from the app's frames (`ui::tests::the_view_overlays_*`
    /// prints them), rounded up.
    fn item(slot: Slot, v: Rect, room: f32) -> Item {
        use Slot::*;
        let f = |s: &[(f32, f32)]| s.iter().map(|&(x, y)| vec2(x, y)).collect::<Vec<_>>();
        // `text` points of text wrapped at `width`: its lines
        let lines = |text: f32, width: f32| (text / width.max(1.0)).ceil().max(1.0);
        let w = v.width();
        match slot {
            // the estimated chip's first line (the state, the source pill, t, step, T) ≈ 560 pt;
            // two help lines of ≈ 420 pt (each wrapped to two at most, else folded)
            Chip => {
                let cw = 506.0f32.min(room).max(224.0);
                let first = 22.0 * lines(480.0, cw - 24.0);
                let note = 16.0 * lines(420.0, cw - 24.0).min(2.0);
                Item::new(slot, f(&[(cw, 16.0 + first + 2.0 * note), (cw, 16.0 + first + 32.0), (cw, 16.0 + first)]))
            }
            Banner => {
                let bw = (w - 120.0).clamp(240.0, 560.0).min(room - 18.0).max(160.0) + 16.0;
                Item::new(slot, f(&[(bw, 16.0 + 18.0 * lines(520.0, bw - 16.0))]))
            }
            CentreTag => Item::new(slot, f(&[(312.0, 52.0)])),
            Capture => Item::new(slot, f(&[(112.0, 40.0)])),
            Segment => Item::new(slot, f(&[(400.0f32.min(w - 40.0) + 22.0, 430.0), (390.0, 40.0)])),
            Offer => Item::new(slot, f(&[(322.0, 112.0)])),
            TeachPrompt => Item::new(slot, f(&[(322.0, 64.0)])),
            AtomCard => Item::new(slot, f(&[(242.0, 342.0), (242.0, 96.0)])).shrinkable(96.0),
            // with the sliders button (the popover's) at its end
            ViewSwitch => Item::new(slot, f(&[(336.0, 46.0)])),
            // one row, two rows, the popover (no room here)
            CameraSliders => Item::new(slot, f(&[(620.0, 26.0), (446.0, 49.0), (0.0, 0.0)])),
            DotsSliders => Item::new(slot, f(&[(742.0, 26.0), (478.0, 49.0), (0.0, 0.0)])),
            OrbitalLegend => {
                let lw = 390.0f32.min((w * 0.45).clamp(200.0, 520.0)) + 22.0;
                Item::new(slot, f(&[(lw, 34.0 + 15.0 * lines(370.0, lw - 22.0).min(2.0)), (lw, 49.0)]))
            }
            KnowledgeLegend => Item::new(slot, f(&[(272.0, 94.0), (272.0, 80.0)])),
            Sampling => Item::new(slot, f(&[(520.0f32.min(w - 40.0) + 22.0, 113.0), (332.0, 37.0)])),
            Relax => Item::new(slot, f(&[(520.0f32.min(w - 40.0) + 22.0, 104.0), (330.0, 40.0)])),
            DensityLabel => {
                let lw = 484.0f32.min((w - 560.0).max(260.0)) + 12.0;
                Item::new(slot, f(&[(lw, 12.0 + 16.0 * lines(484.0, lw - 12.0).min(2.0)), (lw, 30.0)]))
            }
            // at most half the view wide (`overlays::toast`)
            Toast => {
                let tw = 476.0f32.min((w - 28.0) * 0.5 + 56.0);
                Item::new(slot, f(&[(tw, 16.0 + 18.0 * lines(420.0, tw - 56.0))]))
            }
            BondsPair => Item::new(slot, f(&[(104.0, 36.0)])),
            // inside the room the view switch leaves beside it (`overlays::stale_line`)
            StaleLine => {
                let beside = w - 28.0 - 336.0 - GAP;
                let lw = if beside >= 160.0 { 252.0f32.min(beside) } else { 252.0 };
                Item::new(slot, f(&[(lw, 6.0 + 13.0 * lines(400.0, lw - 12.0))]))
            }
            OrbitalsCard => Item::new(slot, f(&[(250.0, 52.0)])),
        }
    }

    const SIZES: [(f32, f32); 4] = [(1920.0, 1080.0), (1186.0, 1280.0), (900.0, 600.0), (560.0, 360.0)];

    /// `slots` laid out in `view` as the app would after a few frames: the chip and the banner
    /// re-measured in the room the last layout left them, until it settles.
    fn settle(view: Rect, slots: &[Slot]) -> (Vec<Item>, ViewLayout) {
        let mut room = view.width() - 2.0 * INSET_TOP;
        let mut last = None;
        for _ in 0..6 {
            let it: Vec<Item> = slots.iter().map(|&s| item(s, view, room)).collect();
            let l = layout(view, &it);
            let settled = (l.top_left_room - room).abs() < 0.01;
            room = l.top_left_room;
            last = Some((it, l));
            if settled {
                break;
            }
        }
        last.expect("laid out")
    }

    fn check(name: &str, view: Rect, slots: &[Slot]) -> ViewLayout {
        let (items, l) = settle(view, slots);
        let items = &items[..];
        let rects: Vec<(Slot, Rect)> = l.rects().collect();
        for (s, r) in &rects {
            assert!(view.contains_rect(*r), "{name} {view:?}: {s:?} {r:?} leaves the view");
        }
        for (k, (a, ra)) in rects.iter().enumerate() {
            for (b, rb) in &rects[k + 1..] {
                assert!(!ra.intersects(*rb), "{name} {view:?}: {a:?} {ra:?} overlaps {b:?} {rb:?}: {:?}", l.placed);
            }
        }
        // every item present is laid out, in one of its forms
        for it in items {
            let p = l.get(it.slot).unwrap_or_else(|| panic!("{name}: {:?} not laid out", it.slot));
            assert!(p.form < it.forms.len());
        }
        assert!(!l.crowded, "{name} {view:?}: crowded");
        l
    }

    fn view(w: f32, h: f32) -> Rect {
        // the view sits under the menus and the toolbar, beside the dock: not at the origin
        Rect::from_min_size(pos2(300.0, 90.0), vec2(w, h))
    }

    /// Every overlay of the view that can show at once (the modal model offer is not in the
    /// zones; while a pass runs the app hides the offers, holds the Orbitals solve and runs no
    /// relax) at the three larger sizes of the spec; at the 560 × 360 minimum, a pass's overlays
    /// without a selection (the segment preview and the atom card come from one, and the atom
    /// card alone is 342 points tall) or the orbital legend: a two-line chip, the ghost tag, the
    /// card, the label, a two-line toast, the sliders' button and the stale line fill the view —
    /// the legend on top of them is the one thing too many (measured: the layout is 30 points
    /// short; the legend fits beside the rest in `the_common_subsets_fit_at_the_four_sizes`).
    #[test]
    fn every_overlay_at_once_fits_at_the_four_sizes() {
        use Slot::*;
        for (w, h) in SIZES {
            let slots: Vec<Slot> = if vec2(w, h) == MIN_VIEW {
                vec![Chip, CentreTag, Capture, ViewSwitch, DotsSliders, CameraSliders, Sampling, DensityLabel, Toast, BondsPair, StaleLine]
            } else {
                Slot::ALL.into_iter().filter(|s| !matches!(s, Offer | TeachPrompt | Relax | OrbitalsCard)).collect()
            };
            check("all", view(w, h), &slots);
        }
    }

    /// The common subsets (spec "Tests and gates"): training, relax, orbitals with the legend,
    /// toasts — each at the four sizes.
    #[test]
    fn the_common_subsets_fit_at_the_four_sizes() {
        use Slot::*;
        let base = [Chip, Capture, ViewSwitch, BondsPair];
        let sets: [(&str, &[Slot]); 7] = [
            ("training", &[Sampling, DensityLabel, CentreTag, StaleLine, Toast]),
            ("relax", &[Relax, DensityLabel, StaleLine, Toast]),
            ("orbitals", &[OrbitalLegend, DensityLabel, OrbitalsCard, StaleLine, DotsSliders]),
            ("camera", &[CameraSliders, DensityLabel, Toast]),
            ("toasts", &[Toast, DensityLabel, StaleLine, Banner]),
            ("knowledge", &[KnowledgeLegend, OrbitalLegend, DensityLabel, Offer]),
            ("selection", &[AtomCard, StaleLine, DensityLabel]),
        ];
        for (name, extra) in sets {
            for (w, h) in SIZES {
                // the minimum view holds the first-pass offer (322 × 112 on the right edge) only
                // without both legends and the label: it is left out of that set there
                let slots: Vec<Slot> = base.iter().chain(extra.iter()).copied().filter(|s| !(*s == Offer && vec2(w, h) == MIN_VIEW)).collect();
                check(name, view(w, h), &slots);
            }
        }
        // the offer with the chip and the switch alone does fit the minimum view (under the chip)
        check("offer", view(MIN_VIEW.x, MIN_VIEW.y), &[Chip, Capture, ViewSwitch, BondsPair, Offer, DensityLabel]);
    }

    /// On a wide view nothing is compacted and bottom-centre stays in the bottom row, centred.
    #[test]
    fn a_wide_view_keeps_the_full_forms_and_the_bottom_row() {
        use Slot::*;
        let v = view(1920.0, 1080.0);
        let l = check("wide", v, &[Chip, Capture, ViewSwitch, BondsPair, StaleLine, Sampling, DensityLabel, Toast, OrbitalLegend]);
        for s in [Sampling, DensityLabel, OrbitalLegend, Chip] {
            assert_eq!(l.form(s), 0, "{s:?} full");
        }
        let card = l.get(Sampling).unwrap().rect;
        assert!((card.bottom() - (v.bottom() - INSET_BOTTOM)).abs() < 0.01, "bottom row: {card:?}");
        assert!((card.center().x - v.center().x).abs() < 0.01, "centred: {card:?}");
        // the stack's order: the card, the label over it, the toast over that, a gap apart
        let (label, toast) = (l.get(DensityLabel).unwrap().rect, l.get(Toast).unwrap().rect);
        assert!((card.top() - label.bottom() - GAP).abs() < 0.01 && (label.top() - toast.bottom() - GAP).abs() < 0.01, "{card:?} {label:?} {toast:?}");
    }

    /// The user's screenshot (1186 wide): the sampling card no longer covers the orbital legend,
    /// the Dots sliders fold onto two rows before anything else moves, and the bottom-right stack
    /// keeps its order (the pair, the stale line just over it, the Orbitals card above).
    #[test]
    fn the_users_1186_view_compacts_the_widest_first() {
        use Slot::*;
        let v = view(1186.0, 1280.0);
        let l = check("1186", v, &[Chip, Capture, ViewSwitch, DotsSliders, OrbitalLegend, Sampling, DensityLabel, BondsPair, StaleLine, OrbitalsCard]);
        assert_eq!(l.form(DotsSliders), 1, "the widest (the sliders) wraps onto two rows first");
        let (pair, line, card) = (l.get(BondsPair).unwrap().rect, l.get(StaleLine).unwrap().rect, l.get(OrbitalsCard).unwrap().rect);
        assert!((pair.top() - line.bottom() - GAP).abs() < 0.01 && (line.top() - card.bottom() - GAP).abs() < 0.01, "{pair:?} {line:?} {card:?}");
        assert!((pair.right() - line.right()).abs() < 0.01 && (pair.right() - card.right()).abs() < 0.01, "right-aligned");
    }

    /// When the bottom row cannot hold bottom-centre even compacted, it moves up into a second
    /// row above the corners — never over them.
    #[test]
    fn bottom_centre_takes_a_second_row_when_the_corners_leave_no_room() {
        use Slot::*;
        let v = view(900.0, 600.0);
        let l = check("900", v, &[Chip, Capture, ViewSwitch, DotsSliders, BondsPair, StaleLine, OrbitalsCard, Sampling]);
        let card = l.get(Sampling).unwrap().rect;
        let corners = [ViewSwitch, DotsSliders, BondsPair, StaleLine, OrbitalsCard].map(|s| l.get(s).unwrap().rect);
        assert!(card.bottom() < v.bottom() - INSET_BOTTOM - 1.0, "moved up: {card:?}");
        assert!(corners.iter().all(|c| !c.intersects(card)), "clear of the corners");
    }

    /// The decision depends on the sizes alone: the same request gives the same layout.
    #[test]
    fn the_layout_is_a_function_of_the_sizes() {
        for (w, h) in SIZES {
            let all: Vec<Item> = Slot::ALL.iter().map(|&s| item(s, view(w, h), 400.0)).collect();
            let (a, b) = (layout(view(w, h), &all), layout(view(w, h), &all));
            assert_eq!(a.placed, b.placed);
        }
    }

    /// The atom card, last on the right edge, scrolls rather than running into the bottom-right
    /// stack; the top-left room stops at the right column beside the chip.
    #[test]
    fn the_atom_card_shrinks_above_the_bottom_right_stack() {
        use Slot::*;
        let v = view(900.0, 420.0);
        let l = check("atom", v, &[Chip, Capture, AtomCard, BondsPair, StaleLine, ViewSwitch]);
        let (card, line) = (l.get(AtomCard).unwrap().rect, l.get(StaleLine).unwrap().rect);
        assert!(card.height() < 342.0 && card.bottom() <= line.top() - GAP + 0.01, "{card:?} {line:?}");
        let capture = l.get(Capture).unwrap().rect;
        let chip_left = v.left() + INSET_TOP;
        assert!((l.top_left_room - (card.left().min(capture.left()) - GAP - chip_left)).abs() < 0.01, "room {}", l.top_left_room);
    }

    /// An overlay not measured yet (zero size) is pinned at its stack's place and takes no room.
    #[test]
    fn an_unmeasured_overlay_is_pinned_at_its_stack() {
        use Slot::*;
        let v = view(900.0, 600.0);
        let mut it: Vec<Item> = [ViewSwitch, OrbitalLegend].iter().map(|&s| item(s, v, 400.0)).collect();
        it[1].forms = vec![Vec2::ZERO, Vec2::ZERO];
        let l = layout(v, &it);
        let (pivot, at) = l.pin(OrbitalLegend).unwrap();
        assert_eq!(pivot, Align2::LEFT_BOTTOM);
        let switch = l.get(ViewSwitch).unwrap().rect;
        assert!((at.y - (switch.top() - GAP)).abs() < 0.01 && (at.x - switch.left()).abs() < 0.01, "{at:?} {switch:?}");
        assert_eq!(l.rects().count(), 1);
    }
}
