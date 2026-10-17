//! What the viewer sends the worker: [`Command`] (with the queue rules — which commands edit the
//! scene, pause, abort a running step, or are superseded by the next one: the parent module doc,
//! "What aborts a running step"), the solver [`Backend`]s, and how the worker applies a command to
//! its driver. Split out of `app/src/sim.rs` in M6 Task 1; the narrative stays in the parent
//! module doc.

use super::*;

#[derive(Clone, Debug)]
pub enum Command {
    /// Replaces everything (preset, file load); the driver pauses, time restarts, the external
    /// field is kept.
    SetScene(Scene),
    /// Teleport (paused only; while playing, drag with `Dyn(DragStart/Move/End)`).
    MoveNucleus {
        index: usize,
        pos: [f64; 3],
    },
    /// Placed at rest; allowed while playing. `a`: the isotope's mass number (`None` = the most
    /// abundant one; an unlisted A is reported as an error and nothing is added).
    AddNucleus {
        z: u8,
        a: Option<u16>,
        pos: [f64; 3],
    },
    /// Allowed while playing.
    RemoveNucleus(usize),
    /// These five pause the run and replace the scene with the driver's current one (positions
    /// and velocities kept) with the setting changed: time, pins, drag and the energy history
    /// restart (M7: a policy switch moves E_KS by tens of Ha per pseudised atom — the
    /// conserved-energy and drift readouts must not span it). The provider is reset, cold.
    SetCharge(i32),
    SetMultiplicity(Option<u32>),
    SetGrid {
        n: usize,
        h: f64,
    },
    SetSigma(Option<f64>),
    /// The ionic policy (M7, `sim::IonicPolicy`): all-electron, pseudopotentials for species with a
    /// core, or for every species.
    SetIonicPolicy(sim::IonicPolicy),
    /// The density shown includes the pseudised species' partial core ρ_pc (M7; a display of the
    /// generator's core model): display only — never an edit, never aborts; from the next evaluation.
    DisplayPartialCore(bool),
    /// Time control, thermostat, field, COM correction, pins and drag springs, passed to the
    /// driver as is. Scene edits go through the variants above (the viewer mirrors those);
    /// `Dyn(ReplaceScene | MoveNucleusDiscontinuous | AddNucleus | RemoveNucleus)` is never sent.
    Dyn(DynCommand),
    /// Estimated solver (`worker` module doc): install the model at `autoload` if the file exists — a
    /// refused load starts nothing (the panel offers to forget it) — else, when the surrogate has
    /// no model and no pass runs, start a sampling pass with `plan`. Ignored by the other
    /// backends.
    EnsureModel {
        autoload: Option<PathBuf>,
        plan: SamplingPlan,
    },
    /// Estimated solver: the surrogate's audit cadence (`FallbackPolicy::audit_every`, ruling
    /// S7: a DFT check every n answered steps; 2000 by default, ruling S14's interim; 0 = never).
    /// The rest of the current policy is kept. Ignored by the other backends.
    SetAudit(usize),
    /// Estimated solver: draft mode (`FallbackPolicy::never_fallback`, `providers::surrogate`,
    /// "Draft mode"): the model answers every step it can be asked at all, extrapolating; the
    /// rest of the policy is kept. Per session, off by default. Ignored by the other backends.
    SetDraft(bool),
    /// Estimated solver, the cloud (`worker` module doc): fetch the isolated-atom profiles for the
    /// viewer's atomic superposition, and ask for a display-only density every `refresh_s`
    /// seconds while playing (`None`: never). Ignored by the other backends.
    SetCloud {
        superposition: bool,
        refresh_s: Option<f64>,
    },
    /// The Activity view's training views are shown (true) or not: the worker builds
    /// `sim::TrainingViews` only while they are (`worker::activity`, "The training views'
    /// schedule"). A view setting: not an edit, never aborts, never reaches the driver.
    SetActivityViews(bool),
    /// The orbital view (orbital-viewer spec §3, §4.3; `worker` module doc, "The orbital view"):
    /// display only — never an edit, never aborts a step; consecutive ones collapse to the last.
    OrbitalView(sim::OrbitalViewRequest),
    Shutdown,
}

impl Command {
    /// Applies the command to the viewer's mirror of the scene: teleports, additions, removals
    /// and settings show at once; the driver's commands (time control, drag springs, pins…)
    /// leave the mirror alone — in particular a drag never jumps the nucleus to the cursor: the
    /// spring moves it, and the mirror follows the snapshots.
    pub fn apply(&self, s: &mut Scene) {
        match self {
            Command::SetScene(new) => *s = new.clone(),
            Command::MoveNucleus { index, pos } => {
                if let Some(n) = s.nuclei.get_mut(*index) {
                    n.pos = *pos;
                    n.vel = [0.0; 3];
                }
            }
            Command::AddNucleus { z, a, pos } => s.nuclei.push(crate::scene::SceneNucleus {
                z: *z,
                a: a.filter(|&a| units::isotope_mass_amu_a(*z, a).is_some()).and_then(|a| crate::scene::canonical_a(*z, a)),
                pos: *pos,
                vel: [0.0; 3],
            }),
            // an isotope change shows at once when the driver will accept it (a listed A); the
            // mass itself lives in the driver's scene and follows with the snapshots
            Command::Dyn(DynCommand::SetIsotope { index, a }) => {
                if let Some(n) = s.nuclei.get_mut(*index) {
                    if units::isotope_mass_amu_a(n.z, *a).is_some() {
                        n.a = crate::scene::canonical_a(n.z, *a);
                    }
                }
            }
            Command::RemoveNucleus(i) => {
                if *i < s.nuclei.len() {
                    s.nuclei.remove(*i);
                }
            }
            Command::SetCharge(c) => s.charge = *c,
            Command::SetMultiplicity(m) => s.multiplicity = *m,
            Command::SetGrid { n, h } => {
                s.grid_n = *n;
                s.grid_h = *h;
            }
            Command::SetSigma(x) => s.sigma = *x,
            Command::SetIonicPolicy(p) => s.ionic = *p,
            Command::DisplayPartialCore(_) => {}
            Command::Dyn(_)
            | Command::EnsureModel { .. }
            | Command::SetAudit(_)
            | Command::SetDraft(_)
            | Command::SetCloud { .. }
            | Command::SetActivityViews(_)
            | Command::OrbitalView(_)
            | Command::Shutdown => {}
        }
    }

    /// Commands the viewer mirrors (scene edits, and the play state: Play / Pause / Step and the
    /// pausing model commands, StartSampling and EnsureModel). The
    /// worker counts them in `Snapshot::edits`; the viewer adopts a snapshot's scene and play
    /// state only once every such command it sent is in it, so a snapshot taken before an edit
    /// never undoes the edit on screen.
    pub fn is_edit(&self) -> bool {
        match self {
            Command::Dyn(c) => matches!(
                c,
                DynCommand::Play | DynCommand::Pause | DynCommand::Step(_) | DynCommand::StartSampling(_) | DynCommand::Relax
            ),
            Command::Shutdown
            | Command::SetAudit(_)
            | Command::SetDraft(_)
            | Command::SetCloud { .. }
            | Command::SetActivityViews(_)
            | Command::DisplayPartialCore(_)
            | Command::OrbitalView(_) => false,
            _ => true,
        }
    }

    /// Whether the driver ends up paused by it (Pause, Step, the scene replacements, and a
    /// sampling pass starting).
    pub fn pauses(&self) -> bool {
        matches!(
            self,
            Command::SetScene(_)
                | Command::SetCharge(_)
                | Command::SetMultiplicity(_)
                | Command::SetGrid { .. }
                | Command::SetSigma(_)
                | Command::SetIonicPolicy(_)
                | Command::EnsureModel { .. }
                | Command::Dyn(DynCommand::Pause | DynCommand::Step(_) | DynCommand::StartSampling(_) | DynCommand::Relax)
        )
    }

    /// Whether the command is handed to the driver (which may refuse it: `last_rejection`); the
    /// worker's own settings (audit cadence, draft mode, the cloud, the model's first teaching, the
    /// Activity view's training views) and Shutdown are not.
    pub(super) fn reaches_driver(&self) -> bool {
        !matches!(
            self,
            Command::EnsureModel { .. }
                | Command::SetAudit(_)
                | Command::SetDraft(_)
                | Command::SetCloud { .. }
                | Command::SetActivityViews(_)
                | Command::Shutdown
        )
    }

    /// Whether a queued copy of this command abandons the step (and solve) in flight (module
    /// doc, R11).
    pub fn aborts(&self) -> bool {
        match self {
            Command::Dyn(c) => matches!(
                c,
                DynCommand::Pause
                    | DynCommand::ReplaceScene(_)
                    | DynCommand::MoveNucleusDiscontinuous { .. }
                    | DynCommand::AddNucleus(_)
                    | DynCommand::RemoveNucleus(_)
                    | DynCommand::StartSampling(_)
                    | DynCommand::CancelSampling
                    | DynCommand::ForgetModel
                    | DynCommand::LoadModel(_)
                    | DynCommand::StopRelax
            ),
            // a policy change applies from the next evaluation on, the cloud, the training views
            // and the orbital view are display only: nothing to abandon
            Command::SetAudit(_)
            | Command::SetDraft(_)
            | Command::SetCloud { .. }
            | Command::SetActivityViews(_)
            | Command::DisplayPartialCore(_)
            | Command::OrbitalView(_) => false,
            _ => true,
        }
    }

    /// The next command makes this one redundant: consecutive drag moves, consecutive teleports
    /// of the same nucleus, consecutive drag pulls (a touchpad's wheel events) — the next one
    /// valid, so a refusal never drops an accepted change; the bookings at fixed positions add
    /// up to the last one's, U(F₂) − U(F₁) + U(F₁) − U(F₀) = U(F₂) − U(F₀).
    pub(super) fn superseded_by(&self, next: Option<&Command>) -> bool {
        match (self, next) {
            (
                Command::Dyn(DynCommand::DragMove { .. }),
                Some(Command::Dyn(DynCommand::DragMove { .. })),
            ) => true,
            (
                Command::Dyn(DynCommand::SetDragForce { .. }),
                Some(Command::Dyn(DynCommand::SetDragForce { f_max })),
            ) => (sim::dynamics::F_DRAG_LO..=sim::dynamics::F_DRAG_HI).contains(f_max),
            (
                Command::MoveNucleus { index: a, .. },
                Some(Command::MoveNucleus { index: b, .. }),
            ) => a == b,
            // the view is a setting: only the last one counts
            (Command::OrbitalView(_), Some(Command::OrbitalView(_))) => true,
            _ => false,
        }
    }
}

/// Which solver the worker's driver runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `providers::CpuKs`: `qm::scf` on the CPU (f64; the reference).
    Cpu,
    /// `providers::GpuKs`: `qm_gpu::GpuSolver` on the viewer's device (f32 grid quantities).
    Gpu,
    /// `providers::SurrogateKs` around a Kohn-Sham teacher: `GpuKs` when a context exists, else
    /// `CpuKs` (`worker` module doc, "The estimated solver").
    Estimated,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Gpu => "GPU",
            Backend::Estimated => "estimated",
        }
    }
}

impl Worker {
    pub(super) fn apply(&mut self, c: Command) {
        match c {
            Command::SetScene(s) => match s.to_sim() {
                Ok(mut new) => {
                    new.external_field = self.dy.snapshot().scene.external_field;
                    self.dy.apply(DynCommand::Pause);
                    self.dy.apply(DynCommand::ReplaceScene(new));
                    // the step count restarts: a new run (`worker::activity`, "Runs")
                    self.activity.new_run();
                }
                Err(e) => self.error(e),
            },
            Command::MoveNucleus { index, pos } => self
                .dy
                .apply(DynCommand::MoveNucleusDiscontinuous { index, pos }),
            Command::AddNucleus { z, a, pos } => match (units::isotope_mass_amu(z), a) {
                (Some(_), None) => self
                    .dy
                    .apply(DynCommand::AddNucleus(sim::Nucleus::new(z, pos))),
                (Some(_), Some(a)) => match sim::Nucleus::isotope(z, a, pos) {
                    Ok(n) => self.dy.apply(DynCommand::AddNucleus(n)),
                    Err(e) => self.error(e),
                },
                (None, _) => self.error(format!("no isotope mass for Z = {z}")),
            },
            Command::RemoveNucleus(i) => self.dy.apply(DynCommand::RemoveNucleus(i)),
            Command::SetCharge(c) => self.replace(|s| s.charge = c),
            Command::SetMultiplicity(m) => self.replace(|s| s.multiplicity = m),
            Command::SetGrid { n, h } => self.replace(|s| s.cell = sim::Cell { n, h }),
            Command::SetSigma(x) => self.replace(|s| s.sigma = x),
            Command::SetIonicPolicy(p) => self.replace(|s| s.ionic = p),
            Command::DisplayPartialCore(on) => self.dy.apply(DynCommand::SetDisplayPartialCore(on)),
            Command::Dyn(c) => {
                // the user's own model actions make the session's model this worker's
                let takes_over = matches!(c, DynCommand::ForgetModel | DynCommand::StartSampling(_) | DynCommand::LoadModel(_));
                let forget = matches!(c, DynCommand::ForgetModel);
                if takes_over {
                    self.pass_offer = None;
                }
                if takes_over || matches!(c, DynCommand::CancelSampling) {
                    // the automatic pass's sizing is for that pass only
                    self.sizing = None;
                }
                self.dy.apply(c);
                if takes_over && self.dy.snapshot().last_rejection.is_none() && trainable(&mut self.dy).is_some() {
                    self.autoload_refused = None;
                    self.owns_autosave = true;
                    if forget {
                        self.last_gap = None;
                    }
                }
            }
            Command::EnsureModel { autoload, plan } => self.ensure_model(autoload, plan),
            Command::SetAudit(n) => {
                self.audit_every = Some(n);
                self.apply_audit();
            }
            Command::SetDraft(on) => {
                self.draft = Some(on);
                self.apply_audit();
            }
            Command::SetCloud { superposition, refresh_s } => {
                self.cloud_superposition = superposition;
                self.cloud_refresh = refresh_s.filter(|x| x.is_finite() && *x > 0.0).map(Duration::from_secs_f64);
            }
            Command::SetActivityViews(on) => self.activity.set_views(on),
            Command::OrbitalView(req) => self.orbital_view(req),
            Command::Shutdown => {} // `Inbox::take` never hands it out
        }
    }

    /// The [`Command::SetAudit`] cadence and [`Command::SetDraft`] onto the surrogate's current
    /// policy.
    pub(super) fn apply_audit(&mut self) {
        let (audit, draft) = (self.audit_every, self.draft);
        if let Some(s) = trainable(&mut self.dy) {
            // (each setter changes the policy only when its value differs)
            if let Some(n) = audit {
                s.set_audit_every(n);
            }
            if let Some(on) = draft {
                s.set_draft(on);
            }
        }
    }

    /// [`Command::OrbitalView`] (module doc, "The orbital view"): to the driver, which forwards it
    /// to its provider and, paused, shows a changed view at once without a solve. In estimated
    /// mode the one producer is the background teacher, whose state is an older geometry's: a
    /// changed view asks it for a display refresh at the current one.
    ///
    /// Cost (Task 3 review M6, for Task 7): *any* changed view asks for that refresh — each new
    /// orbital picked in the list included — even when the side solver's last solve is already at
    /// the current positions: one warm side SCF per click (≈ 0.1–0.3 s at 81³) through a one-job
    /// slot, so fast clicks queue. A "re-display without a solve" job kind would remove it when
    /// the newest side solve's positions equal the scene's; not built.
    pub(super) fn orbital_view(&mut self, req: sim::OrbitalViewRequest) {
        let snap = self.dy.snapshot();
        let changed = !req.same_view(&snap.orbital_view);
        let on = !req.what.is_off();
        self.dy.apply(DynCommand::SetOrbitalView(req));
        if on && changed && !snap.scene.nuclei.is_empty() {
            if let Some(t) = trainable(&mut self.dy) {
                t.request_display(&snap.scene);
            }
        }
    }

    /// Pause, then replace the scene by the driver's current one with `f` applied: a new run
    /// (`worker::activity`, "Runs").
    pub(super) fn replace(&mut self, f: impl FnOnce(&mut sim::Scene)) {
        self.dy.apply(DynCommand::Pause);
        let mut s = self.dy.snapshot().scene;
        f(&mut s);
        self.dy.apply(DynCommand::ReplaceScene(s));
        self.activity.new_run();
    }
}
