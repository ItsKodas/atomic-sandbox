//! The orbital display payloads of the Kohn-Sham adapters (orbital-viewer spec §4.2–4.4 and §7;
//! `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md`; plan Task 3). What `CpuKs` and
//! `GpuKs` (`crate::ks`) share:
//!
//! - [`ViewState`]: the request (`sim::OrbitalViewRequest`), the **rate bound** ([`Cadence`]:
//!   during play at most one payload per `sim::ORBITAL_MIN_INTERVAL` of wall time; forced while
//!   the driver says so, and the first payload after a change or a rebuild always), the
//!   **track ids** the boundary sees, and the **selection fallback**;
//! - the conversion of `qm::orbitals`' rows and verdicts into the boundary's mirrors
//!   (`sim::OrbitalInfo`, …: `sim` does not depend on `qm`, rule 5);
//! - [`Pool`]: display textures that are never overwritten while someone holds them (the
//!   density pool's rule, now one pool per product kind);
//! - [`CpuView`]: the CPU producer, spec §4.2 in f64 with the `qm::orbitals` reference functions
//!   on a `qm::KsState`. The GPU producer is `qm_gpu::orbitals::OrbitalView`.
//!
//! # Track ids across rebuilds
//! A producer (an `OrbitalView` or a [`CpuView`]) is dropped when the view is switched off, on a
//! provider `reset` (a cold re-solve: nuclei added or removed, charge, multiplicity, a replaced
//! scene) and on a grid change (spec §5.4 item 5: the tracker resets, fresh tracks in energy
//! order). Its tracker then counts from 0 again, so the ids handed out are offset by a base that
//! moves past every id exported so far: **an id never names two orbitals** in one provider's
//! lifetime, and a selection made before the rebuild cannot silently pick an unrelated orbital.
//!
//! # The selection fallback (spec §7, "Orbital count changes")
//! `OrbitalRequest::Orbital` names a track. When this update has no row with it (the producer was
//! rebuilt, or the orbital stopped matching), the orbital shown is the one at the same channel
//! and index as the vanished track had in the previous payload, if that exists, else the HOMO
//! (the highest occupied row of either channel) — with the note "selected orbital no longer
//! exists — showing …". The payload's `what` names the orbital actually shown. It is resolved
//! inside the update (`qm_gpu::orbitals::OrbitalView::update_with`), so it costs no second
//! readback.
use gpu::{ColourVolume3D, GpuContext, Volume3D};
use nalgebra::DMatrix;
use qm::orbitals as qo;
use qm::{Grid, KsState};
use sim::{
    Cell, LoneAtomLabels, OrbitalDisplay, OrbitalField, OrbitalInfo, OrbitalLabel, OrbitalRequest, OrbitalViewRequest, Spin, SpinSummary,
    ORBITAL_MIN_INTERVAL,
};
use std::sync::Arc;
use std::time::Instant;

/// The 12 display colours of `qm::orbitals`' colour slots, used by both producers' colour
/// volumes (`qm_gpu::orbitals::ORBITAL_PALETTE`, the one copy).
pub use qm_gpu::orbitals::ORBITAL_PALETTE;

/// `qm::orbitals::Spin` at the boundary.
pub fn spin_out(s: qo::Spin) -> Spin {
    match s {
        qo::Spin::Up => Spin::Up,
        qo::Spin::Down => Spin::Down,
    }
}

/// The boundary's `Spin` as `qm::orbitals::Spin`.
pub fn spin_in(s: Spin) -> qo::Spin {
    match s {
        Spin::Up => qo::Spin::Up,
        Spin::Down => qo::Spin::Down,
    }
}

/// `qm::orbitals::LoneAtom` at the boundary, with the reason for `#k` labels as text (rule 7).
pub fn lone_atom_out(l: qo::LoneAtom) -> LoneAtomLabels {
    match l {
        qo::LoneAtom::NotLoneAtom => LoneAtomLabels::NotLoneAtom,
        qo::LoneAtom::Labelled { shell_order_inverted } => LoneAtomLabels::Labelled { shell_order_inverted },
        qo::LoneAtom::Unlabelled(why) => LoneAtomLabels::Unlabelled(match why {
            qo::Unlabelled::NoCharacter => "an occupied orbital has no dominant angular character",
            qo::Unlabelled::OpenShell => "an open shell: an incomplete l-set makes the potential non-central for both spin channels",
            qo::Unlabelled::SetTest => "a shell fails the angular-character or rank test",
            qo::Unlabelled::NodesUnresolved => "the radial node counts are unresolved or out of Sturm–Liouville order",
        }),
    }
}

/// `qm::orbitals::SpinSummary` at the boundary, with its display scale.
pub fn spin_summary_out(s: &qo::SpinSummary) -> SpinSummary {
    SpinSummary { n_up: s.n_up, n_dn: s.n_dn, integral: s.integral, abs_integral: s.abs_integral, max_abs: s.max_abs, rho_max: s.rho_max, scale: s.scale() }
}

/// The rate bound (spec §4.4): a payload when forced, when one is due (the first after a change
/// or a rebuild), or when the last one is at least `ORBITAL_MIN_INTERVAL` old.
#[derive(Debug, Default)]
pub struct Cadence {
    last: Option<Instant>,
    due: bool,
}

impl Cadence {
    /// The next [`take`](Self::take) says yes whatever the interval.
    pub fn arm(&mut self) {
        self.due = true;
    }

    /// Whether a payload may be made at `now`; a yes counts as one made.
    pub fn take(&mut self, now: Instant, force: bool) -> bool {
        let ok = force || self.due || self.last.is_none_or(|t| now.saturating_duration_since(t) >= ORBITAL_MIN_INTERVAL);
        if ok {
            self.last = Some(now);
            self.due = false;
        }
        ok
    }
}

/// What a `set_orbital_view` changed for the producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewChange {
    /// The same view (at most the rate flag differs): nothing to do.
    Same,
    /// Another view: the next payload is due.
    Changed,
    /// Switched off: drop the producer and its textures.
    Off,
}

/// The product an update makes, chosen from its own rows (module doc, "The selection
/// fallback"); tracks are the producer's raw ids.
#[derive(Clone, Debug, PartialEq)]
pub enum PickKind {
    None,
    Orbital { spin: qo::Spin, track: u32, squared: bool },
    Colour { include: Vec<u32>, physical: bool },
    Spin,
}

/// A resolved request: the product, the request as resolved (boundary ids), and a note.
#[derive(Clone, Debug)]
pub struct Pick {
    pub kind: PickKind,
    pub what: OrbitalRequest,
    pub note: Option<String>,
}

/// The kind of display texture a request needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    None,
    Scalar,
    Colour,
}

/// One provider's orbital-view state (module doc).
#[derive(Debug, Default)]
pub struct ViewState {
    req: OrbitalViewRequest,
    cadence: Cadence,
    /// Added to the producer's raw track ids (module doc, "Track ids across rebuilds").
    base: u32,
    /// One past the largest id exported.
    next: u32,
    /// The last payload's rows (boundary ids): the selection fallback's "same channel and index".
    prev_rows: Vec<OrbitalInfo>,
}

impl ViewState {
    pub fn request(&self) -> &OrbitalViewRequest {
        &self.req
    }

    /// A view is on.
    pub fn on(&self) -> bool {
        !self.req.what.is_off()
    }

    /// Takes the app's request; the caller drops its producer on [`ViewChange::Off`].
    pub fn set(&mut self, req: OrbitalViewRequest) -> ViewChange {
        let changed = !req.same_view(&self.req);
        self.req = req;
        if !self.on() {
            self.rebuilt();
            return ViewChange::Off;
        }
        if changed {
            self.cadence.arm();
            ViewChange::Changed
        } else {
            ViewChange::Same
        }
    }

    /// Whether an evaluation at `now` carries a payload (the rate bound; a yes counts).
    pub fn due(&mut self, now: Instant) -> bool {
        self.on() && self.cadence.take(now, self.req.force_every_step)
    }

    /// `orbital_display_now`: always (when on), and it counts for the rate bound.
    pub fn now(&mut self, now: Instant) -> bool {
        self.on() && self.cadence.take(now, true)
    }

    /// The producer was dropped (off, reset, grid change): later ids start after every id
    /// exported so far, and the next payload is due.
    pub fn rebuilt(&mut self) {
        self.base = self.next;
        self.cadence.arm();
    }

    /// The display texture the request needs.
    pub fn texture(&self) -> TextureKind {
        match self.req.what {
            OrbitalRequest::Orbital { .. } | OrbitalRequest::SpinDensity => TextureKind::Scalar,
            OrbitalRequest::AllColoured { .. } => TextureKind::Colour,
            OrbitalRequest::Off | OrbitalRequest::List => TextureKind::None,
        }
    }

    fn raw(&self, track: u32) -> Option<u32> {
        track.checked_sub(self.base)
    }

    /// Resolves the request against an update's rows (module doc, "The selection fallback").
    pub fn pick(&self, out: &qo::UpdateOutput) -> Pick {
        let export = |t: u32| t + self.base;
        match &self.req.what {
            OrbitalRequest::Off | OrbitalRequest::List => Pick { kind: PickKind::None, what: self.req.what.clone(), note: None },
            OrbitalRequest::SpinDensity => Pick { kind: PickKind::Spin, what: OrbitalRequest::SpinDensity, note: None },
            &OrbitalRequest::Orbital { spin, track, squared } => {
                let found = self.raw(track).and_then(|raw| out.rows.iter().find(|r| spin_out(r.spin) == spin && r.track == raw));
                let (row, note) = match found {
                    Some(r) => (Some(r), None),
                    None => {
                        // the same channel and index as the vanished track had, else the HOMO
                        let same = self.prev_rows.iter().find(|r| r.spin == spin && r.track == track)
                            .and_then(|p| out.rows.iter().find(|r| spin_out(r.spin) == p.spin && r.index == p.index));
                        let homo = || out.rows.iter().filter(|r| r.occupation > 0.0).fold(None::<&qo::Row>, |best, r| match best {
                            Some(b) if b.energy >= r.energy => Some(b),
                            _ => Some(r),
                        });
                        let r = same.or_else(homo);
                        let note = r.map(|r| format!("selected orbital no longer exists — showing {} {}", r.label, if r.spin == qo::Spin::Up { "↑" } else { "↓" }));
                        (r, note)
                    }
                };
                match row {
                    Some(r) => Pick {
                        kind: PickKind::Orbital { spin: r.spin, track: r.track, squared },
                        what: OrbitalRequest::Orbital { spin: spin_out(r.spin), track: export(r.track), squared },
                        note,
                    },
                    None => Pick { kind: PickKind::None, what: self.req.what.clone(), note: None },
                }
            }
            OrbitalRequest::AllColoured { include, physical, additive } => {
                let (raw, note): (Vec<u32>, Option<String>) = match include {
                    None => (out.rows.iter().filter(|r| r.occupation > 0.0).map(|r| r.track).collect(), None),
                    Some(tracks) => {
                        let raw: Vec<u32> = tracks.iter().filter_map(|&t| self.raw(t)).filter(|t| out.rows.iter().any(|r| r.track == *t)).collect();
                        let gone = tracks.len() - raw.len();
                        (raw, (gone > 0).then(|| format!("{gone} included orbital{} no longer exist{}", if gone == 1 { "" } else { "s" }, if gone == 1 { "s" } else { "" })))
                    }
                };
                let what = OrbitalRequest::AllColoured {
                    include: include.as_ref().map(|_| raw.iter().map(|&t| export(t)).collect()),
                    physical: *physical,
                    additive: *additive,
                };
                Pick { kind: PickKind::Colour { include: raw, physical: *physical }, what, note }
            }
        }
    }

    /// The payload of an update (spec §4.3): the rows with boundary ids, the verdicts and notes.
    /// Remembers the rows for the next selection fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn display(&mut self, cell: Cell, origin: [f64; 3], positions: Vec<[f64; 3]>, out: &qo::UpdateOutput, spin: &qo::SpinSummary,
                   pick: Pick, field: Option<OrbitalField>) -> OrbitalDisplay {
        let rows: Vec<OrbitalInfo> = out.rows.iter().map(|r| OrbitalInfo {
            spin: spin_out(r.spin),
            index: r.index,
            track: r.track + self.base,
            slot: r.slot,
            energy: r.energy,
            occupation: r.occupation,
            label: match r.label {
                qo::Label::Index(k) => OrbitalLabel::Index(k),
                qo::Label::Atomic { n, l, m } => OrbitalLabel::Atomic { n, l, m },
            },
            cluster: r.cluster,
            rotated: r.rotated,
            grid_split: r.grid_split,
            box_state: r.box_state,
            smeared_1s: r.smeared_1s,
            residual: None,
        }).collect();
        if let Some(m) = rows.iter().map(|r| r.track + 1).max() {
            self.next = self.next.max(m);
        }
        let mut notes: Vec<String> = pick.note.into_iter().collect();
        match (spin.n_up, spin.n_dn) {
            (0, 0) => notes.push("no electrons — nothing to show".into()),
            (_, 0) => notes.push("↓: none occupied".into()),
            (0, _) => notes.push("↑: none occupied".into()),
            _ => {}
        }
        if self.req.unoccupied > 0 {
            notes.push(format!("{} unoccupied orbitals asked for: not computed yet — only the occupied orbitals are solved", self.req.unoccupied));
        }
        self.prev_rows = rows.clone();
        OrbitalDisplay {
            cell,
            origin,
            positions,
            rows,
            field,
            spin: spin_summary_out(spin),
            unoccupied: None,
            notes,
            what: pick.what,
            lone_atom: lone_atom_out(out.lone_atom),
        }
    }
}

/// A display texture type of a [`Pool`].
pub trait PoolTexture {
    fn create(ctx: &GpuContext, dims: [u32; 3]) -> Self;
    fn dims(&self) -> [u32; 3];
}

impl PoolTexture for Volume3D {
    fn create(ctx: &GpuContext, dims: [u32; 3]) -> Self {
        Volume3D::new(ctx, dims)
    }
    fn dims(&self) -> [u32; 3] {
        self.dims
    }
}

impl PoolTexture for ColourVolume3D {
    fn create(ctx: &GpuContext, dims: [u32; 3]) -> Self {
        ColourVolume3D::new(ctx, dims)
    }
    fn dims(&self) -> [u32; 3] {
        self.dims
    }
}

/// Display textures of one kind. A texture leaves the pool inside a payload as an `Arc` and is
/// written again only when the pool holds the only reference: no result, snapshot or viewer
/// (displayed or bound) still points at it, so a texture is never overwritten while it can be on
/// screen. The check is race-free: other holders can only drop references, never create one from
/// nothing. With a viewer that keeps one payload (plus the crossfade's previous one) this settles
/// at two or three textures; another size drops the old ones.
pub struct Pool<T> {
    pub all: Vec<Arc<T>>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool { all: Vec::new() }
    }
}

impl<T: PoolTexture> Pool<T> {
    pub fn take(&mut self, ctx: &GpuContext, dims: [u32; 3]) -> Arc<T> {
        self.all.retain(|t| t.dims() == dims);
        if let Some(t) = self.all.iter().find(|t| Arc::strong_count(t) == 1) {
            return t.clone();
        }
        let t = Arc::new(T::create(ctx, dims));
        self.all.push(t.clone());
        t
    }
}

/// The CPU producer (spec §4.2 in f64, `qm::orbitals`' reference functions): its tracker and the
/// previous update's displayed orbitals. Display only: it reads the state, never writes it.
pub struct CpuView {
    tracker: qo::Tracker,
    grid: Grid,
    /// The previous update's displayed orbitals per channel, in shown-row order.
    prev: [Vec<Vec<f64>>; 2],
}

impl CpuView {
    pub fn new(grid: &Grid) -> CpuView {
        CpuView { tracker: qo::Tracker::new(), grid: grid.clone(), prev: [Vec::new(), Vec::new()] }
    }

    /// One update on `state` (solved at `nuclei`, SCF tolerance `tol_scf`): the scalars, the
    /// tracker, the displayed orbitals (kept as the next update's previous set) and the product
    /// `view` picks from the rows. Returns the tracker's output, the spin readout, the pick and
    /// the field.
    pub fn update(&mut self, state: &KsState, nuclei: &[qm::Nucleus], tol_scf: f64, view: &ViewState)
                  -> (qo::UpdateOutput, qo::SpinSummary, Pick, Option<OrbitalField>) {
        if state.grid != self.grid {
            *self = CpuView::new(&state.grid);
        }
        let grid = &state.grid;
        let raw: [&Vec<Vec<f64>>; 2] = [&state.orbitals_up, &state.orbitals_dn];
        let n = [raw[0].len(), raw[1].len()];
        let pos: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        let lone = nuclei.len() == 1;
        let refs = |ch: usize| -> Vec<&[f64]> { raw[ch].iter().map(|v| v.as_slice()).collect() };
        let (ms, rp) = if lone {
            ([0, 1].map(|ch| qo::moments_many(grid, &refs(ch), pos[0])), [0, 1].map(|ch| qo::radial_profiles(grid, &refs(ch), pos[0])))
        } else {
            ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()])
        };
        let fs = [0, 1].map(|ch| raw[ch].iter().map(|p| qo::first_sign_functionals(grid, p, &pos)).collect::<Vec<_>>());
        let mx = [0, 1].map(|ch| raw[ch].iter().map(|p| p.iter().fold(0.0f64, |a, v| a.max(v.abs()))).collect::<Vec<_>>());
        let ow = [0, 1].map(|ch| raw[ch].iter().map(|p| qo::outer_weight(grid, p)).collect::<Vec<_>>());
        let prev = &self.prev;
        let overlaps: [Option<DMatrix<f64>>; 2] = [0, 1].map(|ch| {
            (!prev[ch].is_empty()).then(|| DMatrix::from_fn(n[ch], prev[ch].len(), |i, j| grid.dot(&raw[ch][i], &prev[ch][j])))
        });
        let cross = (n[0] > 0 && n[1] > 0).then(|| DMatrix::from_fn(n[0], n[1], |i, j| grid.dot(&raw[0][i], &raw[1][j])));
        let occ = [vec![1.0; n[0]], vec![1.0; n[1]]];
        let input = qo::UpdateInput {
            evals: [&state.evals_up, &state.evals_dn],
            occupations: [&occ[0], &occ[1]],
            overlaps_prev: [overlaps[0].as_ref(), overlaps[1].as_ref()],
            cross: cross.as_ref(),
            moments: lone.then(|| [&ms[0][..], &ms[1][..]]),
            radial: lone.then(|| [&rp[0][..], &rp[1][..]]),
            first_sign: [&fs[0], &fs[1]],
            max_abs: [&mx[0], &mx[1]],
            outer_weight: [&ow[0], &ow[1]],
            n_occ: n,
            n_computed: n,
            n_core: [0; 4],
            z_sigma: lone.then(|| nuclei[0].z as f64 * state.sigma),
            tol_scf,
        };
        let out = self.tracker.update(&input);
        self.prev = [qo::apply_rotation(raw[0], &out, qo::Spin::Up), qo::apply_rotation(raw[1], &out, qo::Spin::Down)];
        let spin = qo::spin_summary(grid, &state.rho_up, &state.rho_dn, n[0], n[1]);
        let pick = view.pick(&out);
        let field = match &pick.kind {
            PickKind::None => None,
            &PickKind::Orbital { spin: s, track, squared } => out.rows.iter().find(|r| r.spin == s && r.track == track).map(|r| {
                let ch = s.index();
                let psi = &self.prev[ch][r.index];
                let sc = out.scale[ch][r.index] as f32;
                if squared {
                    OrbitalField::Cpu { values: Arc::new(psi.iter().map(|v| (v * v) as f32).collect()), signed: false, scale: sc * sc }
                } else {
                    OrbitalField::Cpu { values: Arc::new(psi.iter().map(|&v| v as f32).collect()), signed: true, scale: sc }
                }
            }),
            PickKind::Spin => Some(OrbitalField::Cpu {
                values: Arc::new(state.rho_up.iter().zip(&state.rho_dn).map(|(u, d)| (u - d) as f32).collect()),
                signed: true,
                scale: spin.scale() as f32,
            }),
            PickKind::Colour { include, physical } => {
                let shown: Vec<(&[f64], f64, [f64; 3], f64)> = out.rows.iter().filter(|r| include.contains(&r.track)).filter_map(|r| {
                    let ch = r.spin.index();
                    let pk = if *physical { spin.rho_max } else { out.scale[ch][r.index].powi(2) };
                    if pk <= 0.0 || r.occupation == 0.0 {
                        return None;
                    }
                    let c = ORBITAL_PALETTE[r.slot as usize % ORBITAL_PALETTE.len()];
                    Some((self.prev[ch][r.index].as_slice(), r.occupation, c.map(f64::from), pk))
                }).collect();
                let rgba = qo::colour_volume(grid, &shown).into_iter().map(|v| v.map(|x| x as f32)).collect();
                Some(OrbitalField::CpuColour { rgba: Arc::new(rgba), scale: 1.0 })
            }
        };
        (out, spin, pick, field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The rate bound (spec §4.4), on synthetic instants: the first payload, then one per
    /// `ORBITAL_MIN_INTERVAL`; forced ones always (and they restart the interval); an armed
    /// cadence (a changed view, a rebuild) once regardless. Over 2 s of 5 ms steps: 21 payloads
    /// (t = 0, 0.1, …, 2.0), which is the plan's "≤ 21 in 2 s".
    #[test]
    fn the_cadence_is_one_payload_per_interval_unless_forced_or_armed() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut c = Cadence::default();
        assert!(c.take(at(0), false), "the first");
        assert!(!c.take(at(50), false));
        assert!(!c.take(at(99), false));
        assert!(c.take(at(100), false));
        assert!(c.take(at(101), true), "forced");
        assert!(!c.take(at(150), false), "a forced payload restarts the interval");
        c.arm();
        assert!(c.take(at(160), false), "armed: once");
        assert!(!c.take(at(170), false));
        let mut c = Cadence::default();
        let n = (0..=400).filter(|k| c.take(at(5 * k), false)).count();
        assert_eq!(n, 21);
        let mut c = Cadence::default();
        assert_eq!((0..20).filter(|k| c.take(at(*k), true)).count(), 20, "forced: every one");
    }

    fn row(spin: qo::Spin, index: usize, track: u32, energy: f64, occupation: f64) -> qo::Row {
        qo::Row {
            spin, index, track, slot: 0, energy, occupation, label: qo::Label::Index(index + 1), cluster: None, rotated: false,
            grid_split: None, box_state: false, smeared_1s: false,
        }
    }

    fn output(rows: Vec<qo::Row>) -> qo::UpdateOutput {
        qo::UpdateOutput { rotations: [Vec::new(), Vec::new()], rows, scale: [Vec::new(), Vec::new()], hidden_unoccupied: [0; 2], lone_atom: qo::LoneAtom::NotLoneAtom }
    }

    /// Ids exported past a rebuild never repeat; a selection of a vanished track falls back to the
    /// same channel and index, else to the HOMO, with the note; an included track that vanished is
    /// dropped with a note; "all occupied" takes the occupied rows.
    #[test]
    fn ids_survive_rebuilds_and_a_vanished_selection_falls_back() {
        use qo::Spin::{Down, Up};
        let mut v = ViewState::default();
        assert_eq!(v.set(OrbitalViewRequest::new(OrbitalRequest::List)), ViewChange::Changed);
        let spin = qo::SpinSummary { n_up: 2, n_dn: 1, integral: 1.0, abs_integral: 1.0, max_abs: 0.1, rho_max: 1.0 };
        let first = output(vec![row(Up, 0, 0, -0.6, 1.0), row(Up, 1, 1, -0.3, 1.0), row(Down, 0, 2, -0.55, 1.0)]);
        let p = v.pick(&first);
        let d = v.display(Cell { n: 9, h: 0.5 }, [0.0; 3], Vec::new(), &first, &spin, p, None);
        assert_eq!(d.rows.iter().map(|r| r.track).collect::<Vec<_>>(), [0, 1, 2]);
        // the producer is rebuilt (a reset): raw ids restart at 0, exported ones after 2
        v.rebuilt();
        v.set(OrbitalViewRequest::new(OrbitalRequest::Orbital { spin: Spin::Down, track: 2, squared: false }));
        let second = output(vec![row(Up, 0, 0, -0.62, 1.0), row(Up, 1, 1, -0.31, 1.0), row(Down, 0, 2, -0.5, 1.0), row(Down, 1, 3, -0.2, 1.0)]);
        let p = v.pick(&second);
        assert_eq!(p.kind, PickKind::Orbital { spin: Down, track: 2, squared: false }, "same channel and index (↓ #1), raw id 2");
        assert_eq!(p.what, OrbitalRequest::Orbital { spin: Spin::Down, track: 5, squared: false }, "exported id 3 + 2");
        assert!(p.note.as_deref().is_some_and(|n| n.contains("no longer exists — showing #1 ↓")), "{:?}", p.note);
        let d = v.display(Cell { n: 9, h: 0.5 }, [0.0; 3], Vec::new(), &second, &spin, p, None);
        assert_eq!(d.rows.iter().map(|r| r.track).collect::<Vec<_>>(), [3, 4, 5, 6], "never an old id");
        // a track that is found: shown, no note
        v.set(OrbitalViewRequest::new(OrbitalRequest::Orbital { spin: Spin::Down, track: 6, squared: true }));
        let p = v.pick(&second);
        assert_eq!((p.kind.clone(), p.note.clone()), (PickKind::Orbital { spin: Down, track: 3, squared: true }, None));
        // vanished, and its index gone too (↓ #2 → one ↓ left): the HOMO
        let third = output(vec![row(Up, 0, 7, -0.6, 1.0), row(Up, 1, 8, -0.25, 1.0), row(Down, 0, 9, -0.5, 1.0)]);
        let p = v.pick(&third);
        assert_eq!(p.kind, PickKind::Orbital { spin: Up, track: 8, squared: true }, "the HOMO, ↑ #2 at −0.25");
        // never named at all (an id from another session): the HOMO too
        v.set(OrbitalViewRequest::new(OrbitalRequest::Orbital { spin: Spin::Up, track: 1000, squared: false }));
        assert_eq!(v.pick(&third).kind, PickKind::Orbital { spin: Up, track: 8, squared: false });
        // colour: all occupied, or the listed ones that still exist
        v.set(OrbitalViewRequest::new(OrbitalRequest::AllColoured { include: None, physical: false, additive: true }));
        let mut fourth = third.rows.clone();
        fourth.push(row(Down, 1, 10, 0.1, 0.0));
        let fourth = output(fourth);
        assert_eq!(v.pick(&fourth).kind, PickKind::Colour { include: vec![7, 8, 9], physical: false });
        v.set(OrbitalViewRequest::new(OrbitalRequest::AllColoured { include: Some(vec![2, 3 + 8, 3 + 9]), physical: true, additive: false }));
        let p = v.pick(&fourth);
        assert_eq!(p.kind, PickKind::Colour { include: vec![8, 9], physical: true });
        assert_eq!(p.what, OrbitalRequest::AllColoured { include: Some(vec![11, 12]), physical: true, additive: false });
        assert!(p.note.as_deref().is_some_and(|n| n.contains("1 included orbital no longer exists")), "{:?}", p.note);
        // off: a rebuild; the same view again: nothing
        assert_eq!(v.set(OrbitalViewRequest::default()), ViewChange::Off);
        assert!(!v.on());
        let list = OrbitalViewRequest::new(OrbitalRequest::List);
        assert_eq!(v.set(list.clone()), ViewChange::Changed);
        assert_eq!(v.set(OrbitalViewRequest { force_every_step: true, ..list }), ViewChange::Same);
    }
}
