//! The worker's schedule for Harris display solves (live-orbitals spec §4.2; plan Task 6).
//!
//! In estimated mode with the GPU teacher an orbital view — Orbitals, All orbitals coloured or
//! Spin density, never the Density view nor the bare list — is filled between DFT solves by
//! **Harris display orbitals**: one diagonalisation in the potential of the superposed
//! isolated-atom densities, queued on the background auditor
//! (`ForceProvider::harris_display` → `HarrisAnswer::Queued`; `providers::surrogate`, "Side
//! densities"). This module decides when to ask:
//!
//! - **Eligibility** ([`harris_eligible`]): the estimated solver on the GPU teacher, an orbital
//!   view that draws orbitals ([`harris_view`]), no sampling pass (a pass owns the auditor:
//!   [`HARRIS_PAUSED_PASS`]) and an isolated-atom profile for every species of the scene on its
//!   grid, σ and ionic policy (the cloud's atom solves, one per species: [`HARRIS_WAITING_ATOMS`]
//!   until they are in — `Worker::cloud_jobs` asks for them while such a view is on, even with
//!   the cloud's superposition off). The note, when not eligible under an orbital view, is the
//!   snapshot's `harris_note`.
//! - **When** ([`HarrisSchedule`]): a view change (under Model only) asks at once. Paused, one
//!   solve after each geometry change, [`HARRIS_DEBOUNCE`] after a drag's last move. Playing, at
//!   most one per max([`HARRIS_MIN_INTERVAL`], the View setting "refresh orbitals every N s"
//!   ([`HARRIS_EVERY_DEFAULT`]), last wall / [`HARRIS_BUDGET`]) — at most 15 % of wall time.
//!   A geometry a DFT payload already covers — a converged payload shown there, or a refresh
//!   asked for there by a view change outside Model only — gets no Harris job (the converged one
//!   wins at the same geometry anyway).
//! - **A job that never lands** (Task 5: a cancelled Harris job is silent and never re-sent by
//!   the provider — a newer command raises the display interrupt): a sent job is *awaited* until
//!   a Harris payload at its positions lands. When the background solver is idle again
//!   (`Trainable::side_job_in_flight` false) and it has not landed, or after
//!   [`HARRIS_LAND_TIMEOUT`], the schedule re-arms: the same geometry is due again. At most
//!   [`HARRIS_REARMS`] re-arm per geometry and view, so a payload the driver will not show (a
//!   converged one at the same geometry wins, spec §4.2 "Precedence") never loops.
//! - **Model only** ([`view_change_job`]): a view change asks for a Harris job instead of the DFT
//!   refresh it asks for otherwise (or nothing while Harris is not eligible) — Model only starts
//!   no DFT solve for a view.
//!
//! Display only: nothing here reaches an energy, a force or a sample.

use super::*;
use sim::{HarrisAnswer, HarrisAtoms, OrbitalKind, OrbitalRequest, HARRIS_SAME_GEOMETRY};

/// The most of the wall time Harris display solves may take while playing (spec §4.2).
pub(super) const HARRIS_BUDGET: f64 = 0.15;
/// The shortest interval between two Harris solves while playing.
pub(super) const HARRIS_MIN_INTERVAL: Duration = Duration::from_secs(1);
/// Paused: a solve this long after the last move of a drag (one per drag, not one per frame).
pub(super) const HARRIS_DEBOUNCE: Duration = Duration::from_millis(300);
/// The View setting "refresh orbitals every N s"'s default (`UiState::orbital_refresh_s`).
pub(super) const HARRIS_EVERY_DEFAULT: Duration = Duration::from_secs(2);
/// A sent job not landed after this long is treated as lost and re-armed (module doc). A Harris
/// job waits behind an audit or a refresh at most (seconds at 97³); a false alarm costs one
/// re-sent job, which replaces the queued one (newest wins).
pub(super) const HARRIS_LAND_TIMEOUT: Duration = Duration::from_secs(15);
/// Re-arms per geometry and view (module doc).
pub(super) const HARRIS_REARMS: u32 = 1;

/// Not eligible: the isolated atoms are not all in yet.
pub const HARRIS_WAITING_ATOMS: &str = "orbital display: waiting for the isolated-atom densities (one solve per species, shared with the cloud)";
/// Not eligible: a sampling pass holds the background solver.
pub const HARRIS_PAUSED_PASS: &str = "orbital display solves paused: the sampling pass is using the background solver";
/// Not eligible, and not a note (never published): not the estimated solver on the GPU teacher,
/// a view that draws no orbitals, or a scene with no atoms.
const HARRIS_NOT_HERE: &str = "no Harris display here";

/// The views a Harris solve fills (spec §4.1): a picked orbital, All orbitals coloured and the
/// Spin density. Not the list alone and not off (the Density view asks for `Off`): the default
/// view never starts one.
pub(super) fn harris_view(what: &OrbitalRequest) -> bool {
    matches!(what, OrbitalRequest::Orbital { .. } | OrbitalRequest::AllColoured { .. } | OrbitalRequest::SpinDensity)
}

/// Whether a Harris display solve may be asked for now (module doc, "Eligibility"); the note why
/// not. `profiles`: the isolated-atom profiles held (any key: only those fitting the scene's
/// count).
pub(super) fn harris_eligible(estimated_gpu: bool, what: &OrbitalRequest, scene: &sim::Scene, profiles: &[AtomProfile], pass_active: bool) -> Result<(), &'static str> {
    if !estimated_gpu || !harris_view(what) || scene.nuclei.is_empty() {
        return Err(HARRIS_NOT_HERE);
    }
    if pass_active {
        return Err(HARRIS_PAUSED_PASS);
    }
    let (n, h) = (scene.cell.n, scene.cell.h);
    let missing = scene.nuclei.iter().any(|x| !profiles.iter().any(|p| p.z == x.z && p.fits(n, h, scene.sigma, scene.ionic)));
    if missing {
        return Err(HARRIS_WAITING_ATOMS);
    }
    Ok(())
}

/// What a changed orbital view asks the background solver for ([`view_change_job`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ViewJob {
    /// A display-only DFT refresh at the current geometry (`Trainable::request_display`).
    Refresh,
    /// A Harris display solve.
    Harris,
    Nothing,
}

/// A changed view's job (spec §4.2): outside Model only a DFT refresh, as before; under Model
/// only a Harris solve, or nothing while Harris is not eligible — never a refresh. A view turned
/// off asks for nothing.
pub(super) fn view_change_job(model_only: bool, harris_ok: bool, on: bool) -> ViewJob {
    match (on, model_only, harris_ok) {
        (false, _, _) => ViewJob::Nothing,
        (true, true, true) => ViewJob::Harris,
        (true, true, false) => ViewJob::Nothing,
        (true, false, _) => ViewJob::Refresh,
    }
}

/// Every nucleus within [`HARRIS_SAME_GEOMETRY`] (the same count).
fn same_geometry(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() <= HARRIS_SAME_GEOMETRY)
}

/// When to ask for a Harris solve (module doc, "When" and "A job that never lands").
#[derive(Debug)]
pub(super) struct HarrisSchedule {
    /// The View setting "refresh orbitals every N s".
    every: Duration,
    /// The last landed solve's wall time.
    last_wall: Option<Duration>,
    /// The last job sent: when and at which positions (`None` positions: nothing sent, or
    /// re-armed — the geometry is due again).
    sent_at: Option<Instant>,
    sent_positions: Option<Vec<[f64; 3]>>,
    /// The job sent and not landed yet: when and at which positions.
    awaiting: Option<(Instant, Vec<[f64; 3]>)>,
    /// The positions at the last [`due`](Self::due) and when they last changed (`None`: no move
    /// seen since the schedule started).
    seen: Option<Vec<[f64; 3]>>,
    moved_at: Option<Instant>,
    /// Re-arms since the last move, view change or landing.
    rearms: u32,
    /// Paused, a debounced solve waits until then ([`wake`](Self::wake)).
    wake_at: Option<Instant>,
}

impl Default for HarrisSchedule {
    fn default() -> Self {
        HarrisSchedule { every: HARRIS_EVERY_DEFAULT, last_wall: None, sent_at: None, sent_positions: None, awaiting: None, seen: None, moved_at: None, rearms: 0, wake_at: None }
    }
}

impl HarrisSchedule {
    /// The playing interval: max([`HARRIS_MIN_INTERVAL`], every, last wall / [`HARRIS_BUDGET`]).
    pub(super) fn interval(&self) -> Duration {
        let budget = self.last_wall.map_or(Duration::ZERO, |w| w.div_f64(HARRIS_BUDGET));
        HARRIS_MIN_INTERVAL.max(self.every).max(budget)
    }

    /// Whether a Harris job is due now at `positions` (module doc, "When"). Paused and debouncing,
    /// [`wake`](Self::wake) says when to look again.
    pub(super) fn due(&mut self, now: Instant, paused: bool, positions: &[[f64; 3]], view_changed: bool) -> bool {
        self.wake_at = None;
        match &self.seen {
            Some(s) if same_geometry(s, positions) => {}
            Some(_) => {
                self.seen = Some(positions.to_vec());
                self.moved_at = Some(now);
                self.rearms = 0;
            }
            None => self.seen = Some(positions.to_vec()),
        }
        if view_changed {
            self.rearms = 0;
            return true;
        }
        if self.sent_positions.as_deref().is_some_and(|s| same_geometry(s, positions)) {
            return false;
        }
        if paused {
            match self.moved_at {
                Some(t) if now < t + HARRIS_DEBOUNCE => {
                    self.wake_at = Some(t + HARRIS_DEBOUNCE);
                    false
                }
                _ => true,
            }
        } else {
            self.sent_at.is_none_or(|t| now >= t + self.interval())
        }
    }

    /// A job was queued at `positions`: awaited until it lands.
    pub(super) fn sent(&mut self, now: Instant, positions: &[[f64; 3]]) {
        self.sent_at = Some(now);
        self.sent_positions = Some(positions.to_vec());
        self.awaiting = Some((now, positions.to_vec()));
        self.wake_at = None;
    }

    /// A Harris payload at `positions` landed, solved in `wall`: the budget's measure, and the
    /// awaited job if it is that one.
    pub(super) fn landed(&mut self, wall: Duration, positions: &[[f64; 3]]) {
        self.last_wall = Some(wall);
        if self.awaiting.as_ref().is_some_and(|(_, p)| same_geometry(p, positions)) {
            self.awaiting = None;
            self.rearms = 0;
        }
    }

    /// The geometry is covered by a DFT refresh asked for at `positions` (a view change outside
    /// Model only): no Harris job for it.
    pub(super) fn covered(&mut self, positions: &[[f64; 3]]) {
        self.sent_positions = Some(positions.to_vec());
    }

    /// The awaited job is lost — the background solver is idle (`in_flight` false) and it has not
    /// landed, or [`HARRIS_LAND_TIMEOUT`] passed — and the geometry is re-armed (at most
    /// [`HARRIS_REARMS`] times; module doc). True when it re-armed.
    pub(super) fn check_lost(&mut self, now: Instant, in_flight: bool) -> bool {
        let Some((t, _)) = &self.awaiting else { return false };
        if in_flight && now < *t + HARRIS_LAND_TIMEOUT {
            return false;
        }
        self.awaiting = None;
        if self.rearms >= HARRIS_REARMS {
            return false;
        }
        self.rearms += 1;
        self.sent_positions = None;
        true
    }

    /// The View setting "refresh orbitals every N s".
    pub(super) fn set_every(&mut self, d: Duration) {
        self.every = d;
    }

    /// Paused and debouncing: when the debounced solve comes due (the idle worker's wait).
    pub(super) fn wake(&self) -> Option<Instant> {
        self.wake_at
    }

    /// Not eligible (the view closed, the backend changed, a pass started, the atoms missing):
    /// everything but the setting and the measured wall is forgotten, so the next eligible round
    /// asks at once.
    pub(super) fn reset(&mut self) {
        *self = HarrisSchedule { every: self.every, last_wall: self.last_wall, ..HarrisSchedule::default() };
    }
}

impl Worker {
    /// The estimated solver on the GPU teacher (a background auditor).
    pub(super) fn estimated_gpu(&self) -> bool {
        self.backend == Backend::Estimated && self.engine == Backend::Gpu
    }

    /// [`harris_eligible`] for `what` on `snap`'s scene with the profiles held.
    pub(super) fn harris_ok(&self, what: &OrbitalRequest, snap: &DynSnapshot) -> Result<(), &'static str> {
        harris_eligible(self.estimated_gpu(), what, &snap.scene, &self.profiles, sampling_active(snap))
    }

    /// The Harris display jobs, between ticks (module doc): the note, a landed payload's wall
    /// time, a lost job re-armed, and a due job queued on the background solver.
    pub(super) fn harris_jobs(&mut self, snap: &DynSnapshot) {
        let what = &snap.orbital_view.what;
        let ok = self.harris_ok(what, snap);
        self.harris_note = match ok {
            Err(n) if n != HARRIS_NOT_HERE => Some(n),
            _ => None,
        };
        let view_changed = std::mem::take(&mut self.harris_view_changed);
        // a payload landed (its wall time is the budget's measure)
        if snap.orbitals_seq != self.harris_seen_seq {
            self.harris_seen_seq = snap.orbitals_seq;
            if let Some(o) = &snap.orbitals {
                if let OrbitalKind::Harris { wall_s, .. } = o.kind {
                    if wall_s.is_finite() && wall_s >= 0.0 {
                        self.harris.landed(Duration::from_secs_f64(wall_s), &o.positions);
                    }
                }
            }
        }
        if ok.is_err() {
            self.harris.reset();
            return;
        }
        let now = Instant::now();
        let in_flight = trainable(&mut self.dy).is_some_and(|t| t.side_job_in_flight());
        self.harris.check_lost(now, in_flight);
        let positions: Vec<[f64; 3]> = snap.scene.nuclei.iter().map(|n| n.pos).collect();
        // a converged (DFT) payload shown at this geometry wins over any Harris one (spec §4.2
        // "Precedence"): nothing to solve here
        if snap.orbitals.as_ref().is_some_and(|o| o.kind == OrbitalKind::Converged && same_geometry(&o.positions, &positions)) {
            self.harris.covered(&positions);
        }
        if !self.harris.due(now, snap.paused, &positions, view_changed) {
            return;
        }
        let atoms = self.harris_atoms(&snap.scene);
        // (Phase A: the estimated provider queues; a `Solved` answer is not routed anywhere —
        // Task 3 review minor)
        if let Some(HarrisAnswer::Queued { .. }) = self.dy.provider_mut().harris_display(&snap.scene, &atoms, &mut || false) {
            self.harris.sent(now, &positions);
        }
    }

    /// The profiles a Harris solve superposes: those held for the scene's species on its key,
    /// rebuilt only when the profiles held or the species change.
    fn harris_atoms(&mut self, scene: &sim::Scene) -> Arc<HarrisAtoms> {
        let mut species: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        species.sort_unstable();
        species.dedup();
        if let Some((p, s, a)) = &self.harris_atoms {
            if Arc::ptr_eq(p, &self.profiles) && *s == species {
                return a.clone();
            }
        }
        let (n, h) = (scene.cell.n, scene.cell.h);
        let profiles = self.profiles.iter().filter(|p| species.contains(&p.z) && p.fits(n, h, scene.sigma, scene.ionic)).cloned().collect();
        let atoms = Arc::new(HarrisAtoms { profiles });
        self.harris_atoms = Some((self.profiles.clone(), species, atoms.clone()));
        atoms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::{ColourLook, Spin};

    fn ms(t0: Instant, ms: u64) -> Instant {
        t0 + Duration::from_millis(ms)
    }

    fn at(x: f64) -> Vec<[f64; 3]> {
        vec![[0.0, 0.0, 0.0], [x, 0.0, 0.0]]
    }

    /// Paused: positions change every 50 ms from t = 0 to 500 ms (a drag); the solve is due once,
    /// 300 ms after the last move (t = 800 ms), and not again at that geometry.
    #[test]
    fn debounce_fires_once_after_the_last_move() {
        let t0 = Instant::now();
        let mut s = HarrisSchedule::default();
        // the view opened at the start geometry a second before: sent and landed
        let start = t0 - Duration::from_secs(1);
        assert!(s.due(start, true, &at(1.4), true));
        s.sent(start, &at(1.4));
        s.landed(Duration::from_millis(100), &at(1.4));
        for k in 0..=10u64 {
            let t = ms(t0, 50 * k);
            assert!(!s.due(t, true, &at(1.4 + 0.01 * (k + 1) as f64), false), "dragging at {} ms", 50 * k);
            assert_eq!(s.wake(), Some(ms(t, 300)), "the wake follows the last move");
        }
        let last = at(1.4 + 0.11);
        for t in [600, 700, 799] {
            assert!(!s.due(ms(t0, t), true, &last, false), "{t} ms: still debouncing");
        }
        assert!(s.due(ms(t0, 800), true, &last, false), "800 ms: due");
        s.sent(ms(t0, 800), &last);
        assert_eq!(s.wake(), None);
        for t in [810, 900, 2000] {
            assert!(!s.due(ms(t0, t), true, &last, false), "{t} ms: sent already");
        }
    }

    /// Playing: the interval is max(1 s, every, last wall / 0.15).
    #[test]
    fn playing_rate_follows_budget_and_setting() {
        let mut s = HarrisSchedule::default();
        s.set_every(Duration::from_secs(2));
        s.landed(Duration::from_secs_f64(0.6), &at(1.4));
        assert_eq!(s.interval().as_millis(), 4000, "0.6 s / 0.15");
        s.landed(Duration::from_secs_f64(0.1), &at(1.4));
        assert_eq!(s.interval(), Duration::from_secs(2), "the setting");
        s.set_every(Duration::from_secs(1));
        s.landed(Duration::from_secs_f64(0.05), &at(1.4));
        assert_eq!(s.interval(), Duration::from_secs(1), "the floor");
        // and `due` keeps it: sent at t0, the atoms move every tick
        let t0 = Instant::now();
        let mut s = HarrisSchedule::default();
        s.set_every(Duration::from_secs(2));
        s.landed(Duration::from_secs_f64(0.6), &at(1.4));
        assert!(s.due(t0, false, &at(1.40), false), "nothing sent yet: due");
        s.sent(t0, &at(1.40));
        let mut fired = Vec::new();
        for k in 1..=100u64 {
            let t = ms(t0, 100 * k);
            if s.due(t, false, &at(1.40 + 0.001 * k as f64), false) {
                s.sent(t, &at(1.40 + 0.001 * k as f64));
                fired.push(100 * k);
            }
        }
        assert_eq!(fired, [4000, 8000], "every 4 s while playing");
        // the atoms stand still: nothing new to solve, however long
        let still = at(1.40 + 0.1);
        assert!(s.due(ms(t0, 20_000), false, &still, false));
        s.sent(ms(t0, 20_000), &still);
        assert!(!s.due(ms(t0, 60_000), false, &still, false));
    }

    /// Model only: a changed view asks for a Harris solve, never a DFT refresh (nothing while not
    /// eligible); otherwise a refresh, as before; nothing for a view turned off.
    #[test]
    fn model_only_view_paths_never_request_a_dft_refresh() {
        assert_eq!(view_change_job(true, true, true), ViewJob::Harris);
        assert_eq!(view_change_job(true, false, true), ViewJob::Nothing);
        assert_eq!(view_change_job(false, true, true), ViewJob::Refresh);
        assert_eq!(view_change_job(false, false, true), ViewJob::Refresh);
        for (m, h) in [(true, true), (true, false), (false, true), (false, false)] {
            assert_eq!(view_change_job(m, h, false), ViewJob::Nothing, "off");
            assert_ne!(view_change_job(true, h, true), ViewJob::Refresh);
        }
    }

    fn profile(z: u8, scene: &sim::Scene) -> AtomProfile {
        AtomProfile { z, n: scene.cell.n, h: scene.cell.h, sigma: scene.sigma, ionic: scene.ionic, r: vec![0.0, 1.0], rho: vec![1.0, 0.0], electrons: 1.0 }
    }

    /// Harris runs only for an orbital-drawing view, outside a pass, with every species' profile.
    #[test]
    fn harris_stops_with_the_view_and_the_pass() {
        let scene = crate::scene::h2_r16().to_sim().unwrap();
        let h = [profile(1, &scene)];
        let coloured = OrbitalRequest::AllColoured { include: None, physical: false, look: ColourLook::default() };
        let picked = OrbitalRequest::Orbital { spin: Spin::Up, track: 0, squared: true };
        for what in [&coloured, &picked, &OrbitalRequest::SpinDensity] {
            assert_eq!(harris_eligible(true, what, &scene, &h, false), Ok(()), "{what:?}");
        }
        // the Density view asks for `Off` (`UiState::orbital_request`)
        let density = crate::ui::UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap()).orbital_request().what;
        assert_eq!(density, OrbitalRequest::Off);
        for what in [&OrbitalRequest::Off, &OrbitalRequest::List, &density] {
            assert!(harris_eligible(true, what, &scene, &h, false).is_err(), "{what:?}");
            assert!(!harris_view(what));
        }
        assert_eq!(harris_eligible(true, &coloured, &scene, &h, true), Err(HARRIS_PAUSED_PASS));
        assert!(harris_eligible(false, &coloured, &scene, &h, false).is_err(), "not the estimated GPU solver");
        // a species without its profile: H₂O with H's only, and H's of another grid
        let water = crate::scene::Scene::preset("h2o").unwrap().to_sim().unwrap();
        assert_eq!(harris_eligible(true, &coloured, &water, &[profile(1, &water)], false), Err(HARRIS_WAITING_ATOMS));
        let mut other = profile(1, &scene);
        other.n += 8;
        assert_eq!(harris_eligible(true, &coloured, &scene, &[other], false), Err(HARRIS_WAITING_ATOMS));
        assert_eq!(harris_eligible(true, &coloured, &water, &[profile(1, &water), profile(8, &water)], false), Ok(()));
    }

    /// Task 5's silent cancel (controller ruling): a sent job that never lands — the background
    /// solver went idle without its payload — re-arms the same geometry once, so a paused view
    /// change whose job a later command interrupted still gets its solve; a second loss at that
    /// geometry does not loop; a landed job is not re-sent; a timeout re-arms too.
    #[test]
    fn a_job_that_never_lands_is_asked_again_once() {
        let t0 = Instant::now();
        let p = at(1.4);
        let mut s = HarrisSchedule::default();
        assert!(s.due(t0, true, &p, true), "the view change");
        s.sent(t0, &p);
        assert!(!s.check_lost(ms(t0, 50), true), "still running");
        assert!(!s.due(ms(t0, 50), true, &p, false));
        // a command interrupted it: the solver is idle, nothing landed
        assert!(s.check_lost(ms(t0, 100), false), "lost: re-armed");
        assert!(s.due(ms(t0, 100), true, &p, false), "the same geometry is due again");
        s.sent(ms(t0, 100), &p);
        // lost again (a payload the driver will not show, say): no loop
        assert!(!s.check_lost(ms(t0, 200), false));
        assert!(!s.due(ms(t0, 200), true, &p, false), "at most one re-arm per geometry");
        // a move starts over: sent, landed — nothing to re-arm
        let q = at(1.5);
        assert!(!s.due(ms(t0, 300), true, &q, false), "debouncing");
        assert!(s.due(ms(t0, 600), true, &q, false));
        s.sent(ms(t0, 600), &q);
        s.landed(Duration::from_millis(150), &q);
        assert!(!s.check_lost(ms(t0, 700), false), "landed: nothing lost");
        assert!(!s.due(ms(t0, 700), true, &q, false));
        // an older job's payload does not count for the newer one
        let r = at(1.6);
        assert!(s.due(ms(t0, 1000), true, &r, true));
        s.sent(ms(t0, 1000), &r);
        s.landed(Duration::from_millis(150), &q);
        assert!(s.check_lost(ms(t0, 1100), false), "the job at r never landed");
        // a timeout while the solver stays busy (playing, audits back to back)
        let mut s = HarrisSchedule::default();
        s.sent(t0, &p);
        assert!(!s.check_lost(t0 + HARRIS_LAND_TIMEOUT - Duration::from_millis(1), true));
        assert!(s.check_lost(t0 + HARRIS_LAND_TIMEOUT, true), "timed out: re-armed");
    }

    /// A DFT refresh asked for at a geometry (a view change outside Model only) covers it: no
    /// Harris job there, whose payload the converged one would beat anyway.
    #[test]
    fn a_refresh_covers_its_geometry() {
        let t0 = Instant::now();
        let mut s = HarrisSchedule::default();
        s.covered(&at(1.4));
        assert!(!s.due(t0, true, &at(1.4), false));
        assert!(!s.due(t0, false, &at(1.4), false));
        s.reset();
        assert!(s.due(t0, true, &at(1.4), false), "reset: due at once");
    }
}
