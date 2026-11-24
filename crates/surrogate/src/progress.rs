//! What a running fit is doing, for the refit picture (spec
//! `docs/superpowers/specs/2026-11-22-refit-visualisation-design.md`, plan
//! `docs/superpowers/plans/2026-11-25-refit-visualisation.md`). Display only: nothing here is read
//! by a fit, and a fit with or without a [`Meter`] computes the same numbers bit for bit.
//!
//! The caller (the provider's refit) installs a [`Meter`] on its own thread for the duration of a
//! refit ([`install`]); the fit's code reports to it through the free functions below, which are
//! no-ops when no meter is installed on the calling thread (every other fit: a view's, a test's).
//! The events are the stages' counters:
//! - [`described`]: samples whose design rows and Gram rows are in the accumulator
//!   (`Accumulator::add_many`, a batch at a time);
//! - [`search_round`]: an outer round of the hyperparameter search starts (the spectral search's
//!   and the dense reference search's);
//! - [`decomposition`]: one O(Q′³) decomposition of the Q′ × Q′ matrix finished — a spectrum
//!   (Householder tridiagonalisation and QL, `TwoNoise::spectrum`) or a solve (Cholesky and SPD
//!   inverse, `TwoNoise::solve_large`; a Cholesky of the dense search, `TwoNoise::solve`). The
//!   library kernels inside count as one step each.
//!
//! Only the calling thread is metered: the fit's worker threads report through it (`add_many`
//! counts each batch after its workers joined).
use std::cell::RefCell;
use std::sync::Arc;

/// Which decomposition finished ([`decomposition`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decomposition {
    /// The spectrum of the search's matrix at a noise ratio (tridiagonalisation + QL).
    Spectrum,
    /// A solve at fixed hyperparameters (Cholesky + inverse).
    Solve,
}

/// Receives a running fit's counters (module doc). Implementations store them in relaxed atomics
/// for a display; they must not block.
pub trait Meter: Send + Sync {
    /// `n` more samples of a pass over `total` are in the accumulator.
    fn described(&self, n: usize, total: usize);
    /// Round `round` (1-based) of a hyperparameter search starts, on a matrix of `q` × `q`.
    fn search_round(&self, round: usize, q: usize);
    /// One decomposition of the `q` × `q` matrix finished.
    fn decomposition(&self, kind: Decomposition, q: usize);
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<dyn Meter>>> = const { RefCell::new(None) };
}

/// The meter of the calling thread while it is held; the previous one (normally none) comes back
/// when it drops.
pub struct Installed(Option<Arc<dyn Meter>>);

impl Drop for Installed {
    fn drop(&mut self) {
        let prev = self.0.take();
        CURRENT.with(|c| *c.borrow_mut() = prev);
    }
}

/// Installs `meter` on the calling thread until the guard drops.
pub fn install(meter: Arc<dyn Meter>) -> Installed {
    Installed(CURRENT.with(|c| c.borrow_mut().replace(meter)))
}

fn with(f: impl FnOnce(&dyn Meter)) {
    CURRENT.with(|c| {
        if let Some(m) = c.borrow().as_ref() {
            f(m.as_ref());
        }
    });
}

/// [`Meter::described`] on the calling thread's meter, if any.
pub fn described(n: usize, total: usize) {
    with(|m| m.described(n, total));
}

/// [`Meter::search_round`] on the calling thread's meter, if any.
pub fn search_round(round: usize, q: usize) {
    with(|m| m.search_round(round, q));
}

/// [`Meter::decomposition`] on the calling thread's meter, if any.
pub fn decomposition(kind: Decomposition, q: usize) {
    with(|m| m.decomposition(kind, q));
}

/// Tests: a meter that records every event.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Default)]
pub struct Recorder(pub std::sync::Mutex<Vec<Event>>);

/// An event [`Recorder`] saw.
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Described { n: usize, total: usize },
    Round { round: usize, q: usize },
    Decomposition { kind: Decomposition, q: usize },
}

#[cfg(any(test, feature = "test-hooks"))]
impl Meter for Recorder {
    fn described(&self, n: usize, total: usize) {
        self.0.lock().unwrap().push(Event::Described { n, total });
    }
    fn search_round(&self, round: usize, q: usize) {
        self.0.lock().unwrap().push(Event::Round { round, q });
    }
    fn decomposition(&self, kind: Decomposition, q: usize) {
        self.0.lock().unwrap().push(Event::Decomposition { kind, q });
    }
}

#[cfg(any(test, feature = "test-hooks"))]
impl Recorder {
    /// The events so far.
    pub fn events(&self) -> Vec<Event> {
        self.0.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_meter_sees_its_own_thread_only_and_leaves_with_its_guard() {
        let rec = Arc::new(Recorder::default());
        described(1, 1); // none installed: nothing
        {
            let _g = install(rec.clone());
            described(3, 10);
            search_round(1, 40);
            decomposition(Decomposition::Spectrum, 40);
            // another thread has no meter
            std::thread::spawn(|| described(99, 99)).join().unwrap();
        }
        described(5, 5); // the guard dropped: nothing
        assert_eq!(
            rec.events(),
            vec![Event::Described { n: 3, total: 10 }, Event::Round { round: 1, q: 40 }, Event::Decomposition { kind: Decomposition::Spectrum, q: 40 }]
        );
    }

    #[test]
    fn a_nested_install_restores_the_outer_meter() {
        let (outer, inner) = (Arc::new(Recorder::default()), Arc::new(Recorder::default()));
        let _o = install(outer.clone());
        {
            let _i = install(inner.clone());
            described(1, 2);
        }
        described(2, 2);
        assert_eq!(inner.events(), vec![Event::Described { n: 1, total: 2 }]);
        assert_eq!(outer.events(), vec![Event::Described { n: 2, total: 2 }]);
    }
}
