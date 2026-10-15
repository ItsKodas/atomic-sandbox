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
//!
//! # Unoccupied orbitals (spec §4.5; plan Task 4)
//! With `OrbitalViewRequest::unoccupied` = m > 0 (capped at [`UNOCCUPIED_MAX`]) the producer runs
//! the display eigensolve (`qm::orbitals` §8: `unoccupied_cpu`, or `OrbitalView::unoccupied` on the
//! GPU) on the provider's converged state, never the stepping solve's: the SCF, energy, forces
//! and dynamics are bitwise the same (`unoccupied_solve_leaves_scf_state_untouched`). When it
//! runs ([`UnoccupiedState::plan`], [`UnoccupiedCadence`]):
//! - **paused** (the driver's `force_every_step`: a single step, an edit's re-evaluation, a view
//!   switched on while paused): every payload whose state has no solve yet for this m — one per
//!   step; a re-display of the same state (another mode, another selection) reuses it;
//! - **in play**: at most once per [`UNOCCUPIED_MIN_INTERVAL`], and only while the last solve's
//!   wall ≤ [`UNOCCUPIED_BUDGET`] × the SCF wall summed since it (≤ 10 % of the worker), never
//!   before one has been measured (the first runs paused: a first measurement in play would stall
//!   play for seconds). Task 0: the eigensolve costs 4–50 warm SCFs, so in practice it runs only
//!   when paused at every grid size (`UnoccupiedStatus::paused_only`).
//!
//! Between solves the producer keeps the last vectors and shows them with the occupied rows of
//! the newer state, stamped not `current` with the geometry they were solved at (the tab's
//! "unoccupied from t = … (updated when paused at this grid size)"); a changed m that cannot be
//! solved yet shows none. The rows carry the residuals, the not-converged tag, the box-state tag
//! (with the face-density rule — the surrogate's [`box_affected_face_density`]`(h)`, `qm::orbitals`
//! §8) and occupation 0; the status the block edge's hidden count, the tolerance and the notes.
use crate::surrogate::box_affected_face_density;
use gpu::{ColourVolume3D, GpuContext, Volume3D};
use nalgebra::DMatrix;
use qm::orbitals as qo;
use qm::{Grid, KsState};
use sim::{
    Cell, LoneAtomLabels, OrbitalDisplay, OrbitalField, OrbitalInfo, OrbitalLabel, OrbitalRequest, OrbitalViewRequest, Spin, SpinSummary,
    UnoccupiedStatus, ORBITAL_MIN_INTERVAL,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// During play the unoccupied-orbital eigensolve runs at most once per this much wall time (spec
/// §4.5, `UNOCCUPIED_MIN_INTERVAL`).
pub const UNOCCUPIED_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// …and only while its last wall time is at most this fraction of the SCF wall summed since its
/// previous run (spec §4.5, review SHOULD 11: ≤ 10 % of the worker in play).
pub const UNOCCUPIED_BUDGET: f64 = 0.1;

/// The most unoccupied orbitals per channel a provider solves for (spec §3: "show unoccupied: m",
/// 0–8); a larger request is capped.
pub const UNOCCUPIED_MAX: u8 = 8;

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

/// The unoccupied eigensolve's cadence and budget (spec §4.5; module doc), pure logic on wall
/// times: forced (paused) always; in play at most once per [`UNOCCUPIED_MIN_INTERVAL`] and only
/// while the last solve's wall ≤ [`UNOCCUPIED_BUDGET`] × the SCF wall summed since it; never in
/// play before a solve has been measured.
#[derive(Debug, Default)]
pub struct UnoccupiedCadence {
    last_run: Option<Instant>,
    last_wall: Option<Duration>,
    scf_since: Duration,
}

impl UnoccupiedCadence {
    /// One evaluation's SCF wall (solve and forces), counted toward the budget.
    pub fn add_scf(&mut self, wall: Duration) {
        self.scf_since += wall;
    }

    /// Whether a solve may run at `now`.
    pub fn allows(&self, now: Instant, forced: bool) -> bool {
        forced || (self.last_run.is_some_and(|t| now.saturating_duration_since(t) >= UNOCCUPIED_MIN_INTERVAL) && self.affordable())
    }

    /// The budget clause alone: a measured last wall ≤ `UNOCCUPIED_BUDGET` × the SCF wall since.
    fn affordable(&self) -> bool {
        self.last_wall.is_some_and(|w| w.as_secs_f64() <= UNOCCUPIED_BUDGET * self.scf_since.as_secs_f64())
    }

    /// A solve ran at `now` and took `wall`.
    pub fn ran(&mut self, now: Instant, wall: Duration) {
        self.last_run = Some(now);
        self.last_wall = Some(wall);
        self.scf_since = Duration::ZERO;
    }

    /// The budget keeps the solve to paused steps at the moment: in play it would be refused for
    /// its cost (or because none has been measured), not merely for the interval.
    pub fn paused_only(&self) -> bool {
        !self.affordable()
    }
}

/// What a payload does about the unoccupied rows ([`UnoccupiedState::plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnoccupiedPlan {
    /// Solve now (m, at the current state).
    Solve,
    /// Keep the producer's vectors (this state's, or an earlier one's with the same m).
    Keep,
    /// Drop them: m = 0, or a changed m that cannot be solved yet.
    Clear,
}

/// One provider's unoccupied-orbital bookkeeping (module doc): the cadence, and what the
/// producer's vectors are — for which m, solved at which positions, on the current state or not.
#[derive(Debug, Default)]
pub struct UnoccupiedState {
    cadence: UnoccupiedCadence,
    /// The vectors were solved on the provider's current converged state.
    current: bool,
    /// m of the vectors held (0: none), the positions they were solved at, the last solve's wall,
    /// and the number of solves so far.
    m: usize,
    positions: Vec<[f64; 3]>,
    wall: Duration,
    solves: u64,
}

impl UnoccupiedState {
    /// A new converged state (a successful evaluation; `scf_wall` its solve and forces): the
    /// vectors held are an earlier geometry's now.
    pub fn new_state(&mut self, scf_wall: Duration) {
        self.current = false;
        self.cadence.add_scf(scf_wall);
    }

    /// Whether the producer holds unoccupied vectors (of some m).
    pub fn held(&self) -> bool {
        self.m != 0
    }

    /// The producer and its vectors were dropped (view off, reset, grid change).
    pub fn dropped(&mut self) {
        self.current = false;
        self.m = 0;
        self.positions.clear();
    }

    /// What a payload asking for `m` at `now` does (`forced`: the driver's paused flag).
    pub fn plan(&self, m: usize, now: Instant, forced: bool) -> UnoccupiedPlan {
        if m == 0 {
            return UnoccupiedPlan::Clear;
        }
        if self.current && self.m == m {
            return UnoccupiedPlan::Keep;
        }
        if self.cadence.allows(now, forced) {
            UnoccupiedPlan::Solve
        } else if self.m == m {
            UnoccupiedPlan::Keep
        } else {
            UnoccupiedPlan::Clear
        }
    }

    /// A solve for `m` at `positions` started at `now` and took `wall`.
    pub fn solved(&mut self, m: usize, positions: Vec<[f64; 3]>, now: Instant, wall: Duration) {
        self.cadence.ran(now, wall);
        self.current = true;
        self.m = m;
        self.positions = positions;
        self.wall = wall;
        self.solves += 1;
    }

    /// The payload's status for a request of `requested` (spec §4.5; `sim::UnoccupiedStatus`):
    /// `None` when none were asked for. `info` is the producer's last solve (None: none held);
    /// `out` the update that showed its rows; `extra` further notes (the GPU memory guard).
    pub fn status(&self, requested: u8, info: Option<&qo::Unoccupied>, out: &qo::UpdateOutput, extra: Vec<String>) -> Option<UnoccupiedStatus> {
        if requested == 0 {
            return None;
        }
        let shown = [qo::Spin::Up, qo::Spin::Down].map(|s| out.rows.iter().filter(|r| r.spin == s && r.occupation == 0.0).count());
        let mut notes = Vec::new();
        match info {
            Some(u) if u.rejected.is_some() => notes.push(u.rejected.clone().expect("matched")),
            Some(_) => {
                for (ch, arrow) in [(0usize, "↑"), (1, "↓")] {
                    if out.hidden_unoccupied[ch] > 0 {
                        notes.push(format!("{arrow}: {} requested, {} shown: a degenerate set cut by the block edge is hidden", self.m, shown[ch]));
                    }
                }
                if !self.current {
                    notes.push(if self.cadence.paused_only() {
                        "unoccupied orbitals from an earlier geometry: their eigensolve costs several SCF solves, so it is updated when paused at this grid size".to_string()
                    } else {
                        "unoccupied orbitals from an earlier geometry (updated at most once per second in play)".to_string()
                    });
                }
            }
            None if extra.is_empty() => notes.push("unoccupied orbitals: computed when paused (their eigensolve costs several SCF solves)".to_string()),
            None => {}
        }
        notes.extend(extra);
        Some(UnoccupiedStatus {
            requested,
            shown,
            hidden: out.hidden_unoccupied,
            tol: info.map_or(0.0, |u| u.tol),
            current: self.current && info.is_some_and(|u| u.rejected.is_none()),
            positions: self.positions.clone(),
            paused_only: self.cadence.paused_only(),
            wall_s: self.wall.as_secs_f64(),
            solves: self.solves,
            notes,
        })
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
                   pick: Pick, field: Option<OrbitalField>, unoccupied: Option<UnoccupiedStatus>) -> OrbitalDisplay {
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
            residual: r.residual,
            not_converged: r.not_converged,
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
        self.prev_rows = rows.clone();
        OrbitalDisplay {
            cell,
            origin,
            positions,
            rows,
            field,
            spin: spin_summary_out(spin),
            unoccupied,
            notes,
            what: pick.what,
            lone_atom: lone_atom_out(out.lone_atom),
        }
    }

    /// A payload without rows or field that says why (`note`): the view was refused (the GPU
    /// memory guard, `crate::ks`), so the producer was not built. It still answers the request —
    /// its unoccupied count included (`sim::dynamics` matches a side payload to the view by it).
    pub fn refused(&self, cell: Cell, origin: [f64; 3], positions: Vec<[f64; 3]>, note: String) -> OrbitalDisplay {
        let unoccupied = (self.req.unoccupied > 0).then(|| UnoccupiedStatus { requested: self.req.unoccupied, notes: vec![note.clone()], ..Default::default() });
        OrbitalDisplay {
            cell,
            origin,
            positions,
            rows: Vec::new(),
            field: None,
            spin: SpinSummary::default(),
            unoccupied,
            notes: vec![note],
            what: self.req.what.clone(),
            lone_atom: LoneAtomLabels::NotLoneAtom,
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
    all: Vec<Arc<T>>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool { all: Vec::new() }
    }
}

impl<T> Pool<T> {
    /// Drops the pool's own references (Task 3 review M7: the invariant stays inside the type —
    /// a texture someone still holds lives on in their `Arc` and is never handed out again).
    pub fn clear(&mut self) {
        self.all.clear();
    }

    /// The textures the pool holds (in use or free).
    pub fn len(&self) -> usize {
        self.all.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
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

/// The CPU producer (spec §4.2 in f64, `qm::orbitals`' reference functions): its tracker, the
/// previous update's displayed orbitals and the display eigensolve's unoccupied vectors (module
/// doc, "Unoccupied orbitals"). Display only: it reads the state, never writes it.
pub struct CpuView {
    tracker: qo::Tracker,
    grid: Grid,
    /// The previous update's displayed orbitals per channel, in shown-row order.
    prev: [Vec<Vec<f64>>; 2],
    /// The unoccupied vectors per channel (ascending ε) and the solve that made them.
    unocc: [Vec<Vec<f64>>; 2],
    unocc_info: Option<qo::Unoccupied>,
}

impl CpuView {
    pub fn new(grid: &Grid) -> CpuView {
        CpuView { tracker: qo::Tracker::new(), grid: grid.clone(), prev: [Vec::new(), Vec::new()], unocc: [Vec::new(), Vec::new()], unocc_info: None }
    }

    /// The display eigensolve (`qm::orbitals::unoccupied_cpu`) on `state` (solved at `nuclei`),
    /// warm from the vectors held; they are replaced by the result's (none when rejected).
    pub fn unoccupied(&mut self, state: &KsState, nuclei: &[qm::Nucleus], m: usize, settings: &qm::ScfSettings) -> qo::Unoccupied {
        if state.grid != self.grid {
            *self = CpuView::new(&state.grid);
        }
        let held = self.unocc.iter().any(|v| !v.is_empty());
        let (u, vecs) = qo::unoccupied_cpu(state, nuclei, m, settings, held.then_some(&self.unocc));
        self.unocc = vecs;
        self.unocc_info = Some(u.clone());
        u
    }

    /// Drops the unoccupied vectors (updates show the occupied rows only).
    pub fn clear_unoccupied(&mut self) {
        self.unocc = [Vec::new(), Vec::new()];
        self.unocc_info = None;
    }

    /// The last display eigensolve's result (None before one or after `clear_unoccupied`).
    pub fn unoccupied_info(&self) -> Option<&qo::Unoccupied> {
        self.unocc_info.as_ref()
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
        // the occupied orbitals, then the unoccupied vectors held (module doc)
        let raw: [Vec<&[f64]>; 2] = [
            state.orbitals_up.iter().chain(&self.unocc[0]).map(|v| v.as_slice()).collect(),
            state.orbitals_dn.iter().chain(&self.unocc[1]).map(|v| v.as_slice()).collect(),
        ];
        let n = [raw[0].len(), raw[1].len()];
        let n_occ = [state.orbitals_up.len(), state.orbitals_dn.len()];
        let pos: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        let lone = nuclei.len() == 1;
        let refs = |ch: usize| -> Vec<&[f64]> { raw[ch].clone() };
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
        let info = self.unocc_info.as_ref();
        let evals: [Vec<f64>; 2] = [0, 1].map(|ch| {
            let e = if ch == 0 { &state.evals_up } else { &state.evals_dn };
            e.iter().chain(info.map_or(&[][..], |u| &u.evals[ch][..])).copied().collect::<Vec<f64>>()
        });
        let occ: [Vec<f64>; 2] = [0, 1].map(|ch| (0..n[ch]).map(|i| if i < n_occ[ch] { 1.0 } else { 0.0 }).collect::<Vec<f64>>());
        let input = qo::UpdateInput {
            evals: [&evals[0], &evals[1]],
            occupations: [&occ[0], &occ[1]],
            overlaps_prev: [overlaps[0].as_ref(), overlaps[1].as_ref()],
            cross: cross.as_ref(),
            moments: lone.then(|| [&ms[0][..], &ms[1][..]]),
            radial: lone.then(|| [&rp[0][..], &rp[1][..]]),
            first_sign: [&fs[0], &fs[1]],
            max_abs: [&mx[0], &mx[1]],
            outer_weight: [&ow[0], &ow[1]],
            n_occ,
            n_computed: n,
            n_core: [0; 4],
            z_sigma: lone.then(|| nuclei[0].z as f64 * state.sigma),
            tol_scf,
            unoccupied: info,
            face_limit: box_affected_face_density(grid.h),
        };
        let out = self.tracker.update(&input);
        let owned = |ch: usize| raw[ch].iter().map(|v| v.to_vec()).collect::<Vec<Vec<f64>>>();
        self.prev = [qo::apply_rotation(&owned(0), &out, qo::Spin::Up), qo::apply_rotation(&owned(1), &out, qo::Spin::Down)];
        let spin = qo::spin_summary(grid, &state.rho_up, &state.rho_dn, n_occ[0], n_occ[1]);
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

    /// The unoccupied eigensolve's cadence and budget (spec §4.5, module doc "Unoccupied
    /// orbitals"), on synthetic instants: forced (paused) always; in play never before a measured
    /// solve, then at most once per `UNOCCUPIED_MIN_INTERVAL` and only once the SCF wall since the
    /// last solve is ≥ its wall / `UNOCCUPIED_BUDGET`. Over a long run the summed eigensolve wall
    /// stays ≤ `UNOCCUPIED_BUDGET` × the summed SCF wall + one solve.
    #[test]
    fn the_unoccupied_cadence_runs_paused_and_within_budget_in_play() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let ms = Duration::from_millis;
        let mut c = UnoccupiedCadence::default();
        assert!(!c.allows(at(0), false), "play: none measured yet");
        assert!(c.paused_only());
        assert!(c.allows(at(0), true), "forced: always");
        c.ran(at(0), ms(500));
        assert!(c.paused_only(), "right after a solve no SCF wall has been spent");
        c.add_scf(ms(4000));
        assert!(!c.allows(at(4000), false), "4 s of SCF: 0.1 × 4 s < 0.5 s");
        c.add_scf(ms(1000));
        assert!(c.allows(at(5000), false), "5 s of SCF: 0.5 s ≤ 0.1 × 5 s");
        assert!(!c.paused_only());
        assert!(c.allows(at(5000), true));
        c.ran(at(5000), ms(500));
        c.add_scf(ms(100_000));
        assert!(!c.allows(at(5999), false), "within the interval, whatever the budget");
        assert!(c.allows(at(6000), false));
        // a long play: 20 ms SCF steps, a 300 ms eigensolve whenever allowed
        let mut c = UnoccupiedCadence::default();
        c.ran(at(0), ms(300));
        let (mut t, mut scf, mut eig, mut runs) = (0u64, 0u64, 0u64, 0);
        while t < 600_000 {
            t += 20;
            scf += 20;
            c.add_scf(ms(20));
            if c.allows(at(t), false) {
                c.ran(at(t), ms(300));
                t += 300;
                eig += 300;
                runs += 1;
            }
        }
        println!("10 min of play: {runs} eigensolves, {eig} ms of {scf} ms SCF ({:.3})", eig as f64 / scf as f64);
        assert!(eig as f64 <= UNOCCUPIED_BUDGET * scf as f64 + 300.0);
        // affordable every 300 ms / budget of SCF wall: ⌊scf · budget / 300⌋ runs, less the last partial one
        assert!(runs as f64 >= scf as f64 * UNOCCUPIED_BUDGET / 300.0 - 1.0, "it does run when affordable ({runs})");
        // the plan: solve, keep or clear
        let mut st = UnoccupiedState::default();
        assert_eq!(st.plan(0, at(0), true), UnoccupiedPlan::Clear);
        assert_eq!(st.plan(4, at(0), false), UnoccupiedPlan::Clear, "play, none measured, none held");
        assert_eq!(st.plan(4, at(0), true), UnoccupiedPlan::Solve);
        st.solved(4, vec![[0.0; 3]], at(0), ms(800));
        assert!(st.held());
        assert_eq!(st.plan(4, at(10), true), UnoccupiedPlan::Keep, "the same state and m: no re-solve (a re-display)");
        st.new_state(ms(30));
        assert_eq!(st.plan(4, at(40), true), UnoccupiedPlan::Solve, "paused: one per step");
        st.solved(4, vec![[0.1, 0.0, 0.0]], at(40), ms(800));
        st.new_state(ms(30));
        assert_eq!(st.plan(4, at(2000), false), UnoccupiedPlan::Keep, "play over budget: the earlier geometry's vectors stay");
        assert_eq!(st.plan(2, at(2000), false), UnoccupiedPlan::Clear, "another m that cannot be solved now: none");
        st.dropped();
        assert!(!st.held());
    }

    fn row(spin: qo::Spin, index: usize, track: u32, energy: f64, occupation: f64) -> qo::Row {
        qo::Row {
            spin, index, track, slot: 0, energy, occupation, label: qo::Label::Index(index + 1), cluster: None, rotated: false,
            grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false,
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
        let d = v.display(Cell { n: 9, h: 0.5 }, [0.0; 3], Vec::new(), &first, &spin, p, None, None);
        assert_eq!(d.rows.iter().map(|r| r.track).collect::<Vec<_>>(), [0, 1, 2]);
        // the producer is rebuilt (a reset): raw ids restart at 0, exported ones after 2
        v.rebuilt();
        v.set(OrbitalViewRequest::new(OrbitalRequest::Orbital { spin: Spin::Down, track: 2, squared: false }));
        let second = output(vec![row(Up, 0, 0, -0.62, 1.0), row(Up, 1, 1, -0.31, 1.0), row(Down, 0, 2, -0.5, 1.0), row(Down, 1, 3, -0.2, 1.0)]);
        let p = v.pick(&second);
        assert_eq!(p.kind, PickKind::Orbital { spin: Down, track: 2, squared: false }, "same channel and index (↓ #1), raw id 2");
        assert_eq!(p.what, OrbitalRequest::Orbital { spin: Spin::Down, track: 5, squared: false }, "exported id 3 + 2");
        assert!(p.note.as_deref().is_some_and(|n| n.contains("no longer exists — showing #1 ↓")), "{:?}", p.note);
        let d = v.display(Cell { n: 9, h: 0.5 }, [0.0; 3], Vec::new(), &second, &spin, p, None, None);
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
