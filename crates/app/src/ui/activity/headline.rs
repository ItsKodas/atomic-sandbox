//! The activity headline (spec §3.1): the status chip's state text while work runs, the toolbar
//! bar's mode, and the 100 ms refresh (§6). Built from the latest snapshot and the probe, so a
//! long solve that publishes no snapshot still moves (its elapsed time and SCF iteration).
//!
//! Words: a pass's outer unit is a **sample** — one full DFT calculation at one geometry, one
//! training example ("Sample 42 of 158") — and its inner unit an **iteration**, one SCF
//! refinement round ("iteration 37 / 200": of the cap, since the total is unknown before
//! convergence; a warm solve that failed and is retried cold counts again from 1, "cold retry").
use super::super::PanelInput;
use crate::sim::Backend;
use std::time::{Duration, Instant};

/// The headline is rebuilt at most this often (spec §6).
pub const REFRESH: Duration = Duration::from_millis(100);
/// One sweep of the indeterminate bar.
pub const SWEEP: Duration = Duration::from_millis(1200);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BarMode {
    Fraction(f32),
    Sweep,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Headline {
    /// The chip's state text.
    pub text: String,
    /// The second pill: a background audit alongside.
    pub audit: Option<String>,
    /// A pill for the orbital viewer's display eigensolve (the orbital viewer, `PanelInput::
    /// orbital_waiting`), when something else is the headline.
    pub orbitals: Option<String>,
    pub bar: BarMode,
}

/// A value rebuilt at most every [`REFRESH`].
pub struct Throttled<T> {
    at: Option<Instant>,
    value: T,
}

impl<T> Throttled<T> {
    pub fn new(value: T) -> Throttled<T> {
        Throttled { at: None, value }
    }

    /// The cached value, rebuilt by `f` when [`REFRESH`] has passed since the last build.
    pub fn get(&mut self, now: Instant, f: impl FnOnce() -> T) -> &T {
        if self.at.is_none_or(|a| now.saturating_duration_since(a) >= REFRESH) {
            self.value = f();
            self.at = Some(now);
        }
        &self.value
    }

    /// The cached value, without refreshing it.
    pub fn peek(&self) -> &T {
        &self.value
    }
}

/// The running solve's iteration: "iteration 7 / 200" from the solver's own record (`live`, of
/// the cap; "· cold retry" on the providers' second attempt), else the probe's step count
/// "iteration 7"; with the last solve's count when one finished ("(last solve 11)").
pub fn steps_text(live: Option<&sim::ScfLive>, step: u32, last: Option<u32>) -> String {
    let now = iteration_text(live, step);
    match last {
        Some(n) => format!("{now} (last solve {n})"),
        None => now,
    }
}

/// "iteration 37 / 200" (· cold retry), or "iteration 37" without the solver's record.
pub fn iteration_text(live: Option<&sim::ScfLive>, step: u32) -> String {
    match live.filter(|l| l.iteration > 0) {
        Some(l) => {
            let retry = if l.attempt > 1 { " · cold retry" } else { "" };
            format!("iteration {} / {}{retry}", l.iteration, l.max_iter)
        }
        None => format!("iteration {step}"),
    }
}

/// The slot's iteration while it runs, "" when it has none to show (the CPU solver without an
/// observer reads 0: say nothing rather than "iteration 0").
fn running_iteration(slot: &sim::activity::SlotView) -> Option<String> {
    let has = slot.live.is_some_and(|l| l.iteration > 0) || slot.scf_step > 0;
    (slot.running && has).then(|| iteration_text(slot.live.as_ref(), slot.scf_step))
}

/// "Sample 42 of 158": the sample being solved (done + 1, at most the total).
pub fn sample_text(done: usize, total: usize) -> String {
    format!("Sample {} of {total}", (done + 1).min(total.max(1)))
}

/// The phase (0..1) of the indeterminate bar at `secs` seconds.
pub fn sweep_phase(secs: f64) -> f32 {
    ((secs / SWEEP.as_secs_f64()).fract()) as f32
}

/// Left edge (a fraction of the bar's width, −0.2..1) of the indeterminate bar's 20 % segment at
/// the phase `p`.
pub fn sweep_edge(p: f32) -> f32 {
    p * 1.2 - 0.2
}

/// "1:40" for 100 s.
fn clock(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// The headline for what runs now (spec §3.1; first match wins), or `None` when nothing does (the
/// chip then keeps its old idle texts). Bond references alone give a headline with no text: the
/// bar shows them (spec §3.2) and the chip keeps its dim note line.
pub fn headline(input: &PanelInput<'_>, probe: Option<&sim::ProbeView>) -> Option<Headline> {
    let p = probe.copied().unwrap_or_default();
    let orbitals = input.orbital_waiting.then(|| "Computing orbitals".to_string());
    let Some(snap) = input.snapshot else {
        return orbitals.map(|o| Headline { text: format!("{o}…"), audit: None, orbitals: None, bar: BarMode::Sweep });
    };
    let d = &snap.dynamics;
    let audit = input
        .model()
        .filter(|m| m.audit_in_flight && p.side.running)
        .map(|_| match running_iteration(&p.side) {
            Some(it) => format!("Audit running · {it}"),
            None => "Audit running".to_string(),
        });
    let last = (d.eval.scf_steps > 0).then_some(d.eval.scf_steps as u32);
    let primary: Option<(String, BarMode)> = if let Some(s) = d.sampling.as_ref().filter(|_| input.sampling()) {
        let frac = BarMode::Fraction(if s.total > 0 { s.done as f32 / s.total as f32 } else { 0.0 });
        // the stage of the job being solved now (the ghost's), not the last finished one's
        let stage = super::words::stage_word(d.ghost.as_ref().map_or(s.stage, |g| g.stage));
        let step = running_iteration(&p.main).map(|it| format!(" · {it}")).unwrap_or_default();
        let eta = input.model().and_then(|m| m.pass_eta_s).map(|e| format!(" · {} left", clock(e))).unwrap_or_default();
        Some((format!("{} · {stage}{eta}{step}", sample_text(s.done, s.total)), frac))
    } else if let Some((t, n)) = p.refit {
        let secs = if t >= 0.3 { format!(" · {t:.1} s") } else { String::new() };
        Some((format!("Refitting · {n} samples{secs}"), BarMode::Sweep))
    } else if p.main.running {
        if snap.engine == Backend::Cpu && running_iteration(&p.main).is_none() {
            Some((format!("Solving (CPU reference) · {:.0} s", p.main.elapsed_s), BarMode::Sweep))
        } else {
            let who = if snap.backend == Backend::Estimated { "DFT fallback" } else { "Solving" };
            Some((format!("{who} · {}", steps_text(p.main.live.as_ref(), p.main.scf_step, last)), BarMode::Sweep))
        }
    } else if let Some(r) = d.relax.as_ref().filter(|r| r.converged.is_none()) {
        Some((format!("Relaxing (FIRE) · step {}", r.steps), BarMode::Sweep))
    } else {
        None
    };
    // bond references: background work whose line stays the chip's dim note; only the bar shows it
    let bond = d.bond_reference.map(|r| BarMode::Fraction(if r.total > 0 { r.done as f32 / r.total as f32 } else { 0.0 }));
    match primary {
        Some((text, bar)) => Some(Headline { text, audit, orbitals, bar }),
        // a background audit beside a model-answered run: "playing" with the second pill; paused,
        // the audit is the headline
        None => match (audit, orbitals) {
            (Some(a), o) if input.playing => Some(Headline { text: "playing".into(), audit: Some(a), orbitals: o, bar: BarMode::Sweep }),
            (Some(a), o) => Some(Headline { text: a, audit: None, orbitals: o, bar: BarMode::Sweep }),
            (None, Some(o)) => Some(Headline { text: format!("{o}…"), audit: None, orbitals: None, bar: BarMode::Sweep }),
            (None, None) => bond.map(|bar| Headline { text: String::new(), audit: None, orbitals: None, bar }),
        },
    }
}

/// [`headline`] as the chrome shows it: while the pass is in its references stage the text names
/// the last reference that *finished* ("Sample 4 of 273 · reference (last done: H) · iteration 9"),
/// from this pass's newest reference row (`ActivityState::current_reference`). It does not claim
/// to name the species being solved: a reference row lands when its solve completes.
pub fn display(input: &PanelInput<'_>, reference: Option<u8>) -> Option<Headline> {
    let mut h = headline(input, input.probe.as_ref())?;
    let in_references = input.snapshot.and_then(|s| s.dynamics.sampling.as_ref()).is_some_and(|p| p.stage == "references") && input.sampling();
    if let (Some(z), true) = (reference, in_references) {
        h.text = h.text.replacen(" · reference", &format!(" · reference (last done: {})", units::isotopes::symbol(z)), 1);
    }
    Some(h)
}
