//! "Bonds now" by itself after a drag (user request 2026-11-13): under the estimated solver the
//! surrogate has no electrons, so the bonds on screen are an earlier DFT step's (`bonds_stale`)
//! and a dragged atom leaves them behind. When a drag of an atom (Select) or of a selection (Edit's
//! handles) ends and nothing is dragged for [`DELAY`], one `Action::RefreshBonds` is sent — the
//! same teacher solve as the bottom-right button, through `ui::Action`, so `--record` sees it. A
//! new drag restarts the wait. View ▸ Bonds & labels ▸ "refresh bonds after a drag"
//! (`UiState::auto_refresh_bonds`, on by default, display only) turns it off.
//!
//! It fires only under the estimated solver with a fitted model, outside a sampling pass, with
//! the bonds shown, at least two atoms, and the bonds stale (or none) when the wait is over. A
//! refresh already pending or running, or the bond references still being solved, hold it until
//! they are done (at most [`GIVE_UP`] past its time); a fresh graph then drops it, unless the
//! worker is still evaluating the moved atoms (its snapshot may be the one from before the drag).
//! Timing and display only: nothing here changes the physics.
use super::{PanelInput, UiState};
use crate::sim::Backend;
use std::time::{Duration, Instant};

/// The wait after the last drag ends.
pub const DELAY: Duration = Duration::from_millis(1000);
/// How long past its time a held refresh may still fire.
pub const GIVE_UP: Duration = Duration::from_secs(20);
/// How often a held refresh looks again.
const RECHECK: Duration = Duration::from_millis(100);

/// The debounce: when the refresh is due (`None`: nothing armed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutoBonds {
    due: Option<Instant>,
    /// The `RefreshBonds` in the frame's actions is this one's (not the button's): its apply leaves
    /// the lengths and angles toggles as the user set them.
    pub fired: bool,
}

/// What the frame's state says about an armed refresh (`gate`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gate {
    /// View ▸ Bonds & labels ▸ refresh bonds after a drag, with the bonds shown.
    pub on: bool,
    /// The estimated solver with a fitted model, no sampling pass, at least two atoms.
    pub surrogate: bool,
    /// The bonds on screen are an earlier DFT step's, or there are none.
    pub stale: bool,
    /// A "Bonds now" is pending or running, or the bond references are still being solved.
    pub held: bool,
    /// The worker is evaluating (a drag's last move may not have reached the snapshot yet).
    pub busy: bool,
}

/// What [`AutoBonds::poll`] says to do this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poll {
    /// Nothing armed (or it was dropped).
    Idle,
    /// Look again in this long (ask for a repaint then).
    Wait(Duration),
    /// Send `Action::RefreshBonds` now (once: the debounce is disarmed).
    Fire,
}

impl AutoBonds {
    /// A drag began: whatever was armed waits for this one to end.
    pub fn drag_started(&mut self) {
        self.due = None;
    }

    /// A drag that moved something ended at `now`: the refresh is due [`DELAY`] later.
    pub fn drag_ended(&mut self, now: Instant) {
        self.due = Some(now + DELAY);
    }

    /// A refresh is armed.
    #[cfg(test)]
    pub fn armed(&self) -> bool {
        self.due.is_some()
    }

    /// The frame at `now` under `g` (module doc).
    pub fn poll(&mut self, now: Instant, g: Gate) -> Poll {
        let Some(due) = self.due else { return Poll::Idle };
        if !g.on || !g.surrogate {
            self.due = None;
            return Poll::Idle;
        }
        if now < due {
            return Poll::Wait(due - now);
        }
        if now > due + GIVE_UP {
            self.due = None;
            return Poll::Idle;
        }
        if g.held || (!g.stale && g.busy) {
            return Poll::Wait(RECHECK);
        }
        self.due = None;
        if g.stale {
            Poll::Fire
        } else {
            Poll::Idle
        }
    }
}

/// The gate of this frame's state.
pub fn gate(st: &UiState, input: &PanelInput<'_>) -> Gate {
    let snap = input.snapshot.filter(|s| s.backend == Backend::Estimated);
    let d = snap.map(|s| &s.dynamics);
    Gate {
        on: st.auto_refresh_bonds && st.show_bonds,
        surrogate: st.backend == Backend::Estimated
            && input.model().is_some_and(|m| m.fitted)
            && !input.sampling()
            && snap.is_some_and(|s| s.scene.nuclei.len() >= 2),
        stale: d.is_some_and(|d| d.bonds_stale || d.bonds.is_none()),
        held: st.bonds_refresh.is_some() || d.is_some_and(|d| !d.bonds_pending.is_empty()),
        busy: input.busy,
    }
}

/// One frame at `now`: polls the debounce under this frame's gate; on `Fire` marks the action as
/// this one's ([`AutoBonds::fired`]) and returns it.
pub fn step(st: &mut UiState, input: &PanelInput<'_>, now: Instant) -> Poll {
    let g = gate(st, input);
    let p = st.auto_bonds.poll(now, g);
    if p == Poll::Fire {
        st.auto_bonds.fired = true;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: Gate = Gate { on: true, surrogate: true, stale: true, held: false, busy: false };

    #[test]
    fn it_fires_once_after_the_delay() {
        let t0 = Instant::now();
        let mut a = AutoBonds::default();
        assert_eq!(a.poll(t0, LIVE), Poll::Idle, "nothing armed: no drag yet");
        a.drag_ended(t0);
        assert_eq!(a.poll(t0, LIVE), Poll::Wait(DELAY));
        assert!(matches!(a.poll(t0 + Duration::from_millis(999), LIVE), Poll::Wait(w) if w == Duration::from_millis(1)));
        assert_eq!(a.poll(t0 + DELAY, LIVE), Poll::Fire);
        assert!(!a.armed());
        assert_eq!(a.poll(t0 + DELAY + Duration::from_millis(16), LIVE), Poll::Idle, "once");
        assert_eq!(a.poll(t0 + 3 * DELAY, LIVE), Poll::Idle);
    }

    #[test]
    fn a_new_drag_restarts_the_wait() {
        let t0 = Instant::now();
        let mut a = AutoBonds::default();
        a.drag_ended(t0);
        let t1 = t0 + Duration::from_millis(700);
        a.drag_started();
        assert_eq!(a.poll(t0 + DELAY, LIVE), Poll::Idle, "a drag in progress: nothing due");
        a.drag_ended(t1 + Duration::from_millis(500));
        let end = t1 + Duration::from_millis(500);
        assert!(matches!(a.poll(t0 + DELAY, LIVE), Poll::Wait(_)), "the first drag's time has passed: still waiting");
        assert!(matches!(a.poll(end + Duration::from_millis(990), LIVE), Poll::Wait(_)));
        assert_eq!(a.poll(end + DELAY, LIVE), Poll::Fire, "a second after the second drag");
        assert_eq!(a.poll(end + 2 * DELAY, LIVE), Poll::Idle, "one refresh for both drags");
    }

    #[test]
    fn it_does_not_fire_off_fresh_or_without_the_surrogate() {
        let t0 = Instant::now();
        let at = t0 + DELAY;
        for (g, why) in [
            (Gate { on: false, ..LIVE }, "the setting is off"),
            (Gate { surrogate: false, ..LIVE }, "DFT, or no model"),
            (Gate { stale: false, ..LIVE }, "the bonds are fresh"),
        ] {
            let mut a = AutoBonds::default();
            a.drag_ended(t0);
            assert_eq!(a.poll(at, g), Poll::Idle, "{why}");
            assert!(!a.armed(), "{why}: dropped, not held");
            assert_eq!(a.poll(at + DELAY, LIVE), Poll::Idle, "{why}: never later either");
        }
    }

    #[test]
    fn a_running_refresh_holds_it_and_a_busy_worker_defers_the_fresh_check() {
        let t0 = Instant::now();
        let at = t0 + DELAY;
        let mut a = AutoBonds::default();
        a.drag_ended(t0);
        assert_eq!(a.poll(at, Gate { held: true, ..LIVE }), Poll::Wait(RECHECK), "a refresh pending or running: not now");
        assert_eq!(a.poll(at + Duration::from_secs(2), LIVE), Poll::Fire, "and once it is done, if the bonds are still stale");
        // fresh but the worker is still on the drag's last move: wait for its snapshot
        let mut a = AutoBonds::default();
        a.drag_ended(t0);
        assert_eq!(a.poll(at, Gate { stale: false, busy: true, ..LIVE }), Poll::Wait(RECHECK));
        assert_eq!(a.poll(at + RECHECK, LIVE), Poll::Fire);
        // held for too long: dropped
        let mut a = AutoBonds::default();
        a.drag_ended(t0);
        assert_eq!(a.poll(at + GIVE_UP + Duration::from_millis(1), LIVE), Poll::Idle);
        assert!(!a.armed());
    }
}
