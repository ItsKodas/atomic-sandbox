//! What the chrome draws over the 3D view (the rect the panels leave): the status chip
//! (top-left: state, who answered, time, step, T, thermostat; red with the error when there is
//! one), the atom card (top-right), the long first-pass offer (top-right, under the card), the
//! sampling card (bottom centre, while a pass runs), the faded-density label (bottom centre),
//! toasts (bottom-right) and the Volume / Isosurface switch with the axes gizmo (bottom-left).
//! The distance guides, drag spring and pin squares stay with the viewer (`viewer.rs`), which
//! projects them from the camera; the cell, floor, axes and placement plane are 3D lines
//! (`overlay.rs`).
use super::dock;
use super::activity;
use super::{source_colour, source_label, theme, Action, PanelInput, UiState, GREEN, RED, SPARK_E};
use crate::sim::Backend;
use egui::{pos2, Align2, Color32, Id, Order, Pos2, Rect, RichText, Stroke, Vec2};

pub fn show(ctx: &egui::Context, view: Rect, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if let Some(s) = input.snapshot {
        st.temp_avg.observe(ctx.input(|i| i.time), &s.dynamics);
    }
    if view.width() < 50.0 || view.height() < 50.0 {
        return;
    }
    let chip = status_chip(ctx, view, st, input);
    bonds_now(ctx, view, st, input, acts);
    let mut right_y = view.top() + 12.0;
    if st.show_atom_card {
        if let Some((i, n)) = input.selection() {
            let pseudo = input.snapshot.and_then(|s| super::pseudo::of_species(&s.dynamics.eval.pseudised, n.z));
            let r = crate::atom_card::show(ctx, pos2(view.right() - 12.0, right_y), n, i, pseudo);
            right_y = r.bottom() + 10.0;
        }
    }
    if let Some(o) = input.model().and_then(|m| m.pass_offer.as_ref()).filter(|_| !input.sampling()) {
        egui::Area::new(Id::new("offer-card"))
            .pivot(Align2::RIGHT_TOP)
            .fixed_pos(pos2(view.right() - 12.0, right_y))
            .order(Order::Middle)
            .fade_in(false)
            .show(ctx, |ui| {
                theme::card().stroke(Stroke::new(1.0, super::AMBER.gamma_multiply(0.5))).show(ui, |ui| {
                    ui.set_max_width(300.0);
                    dock::offer(ui, o, acts);
                });
            });
    }
    model_offer_card(ctx, view, st, input, acts);
    // the teach-around prompt: one click, never run by itself
    if let Some(m) = input.model().filter(|_| teach_prompt(input).is_some()) {
        egui::Area::new(Id::new("teach-prompt-card"))
            .pivot(Align2::RIGHT_TOP)
            .fixed_pos(pos2(view.right() - 12.0, right_y))
            .order(Order::Middle)
            .fade_in(false)
            .show(ctx, |ui| {
                theme::card().stroke(Stroke::new(1.0, super::AMBER.gamma_multiply(0.5))).show(ui, |ui| {
                    ui.set_max_width(300.0);
                    dock::teach_row(ui, m, input.sampling(), acts);
                });
            });
    }
    let mut bottom = view.bottom() - 14.0;
    // in an orbital mode the orbitals' own label or note replaces the density's
    let layer_label = if st.show == super::ShowMode::Density { &input.density_label } else { &input.orbital_label };
    let label = match (layer_label, &input.bonds_label) {
        (Some(d), Some(b)) => Some(format!("{d}\n{b}")),
        (d, b) => d.clone().or_else(|| b.clone()),
    };
    if let Some(label) = &label {
        let r = egui::Area::new(Id::new("density-label"))
            .pivot(Align2::CENTER_BOTTOM)
            .fixed_pos(pos2(view.center().x, bottom))
            .interactable(false)
            .order(Order::Middle)
            .fade_in(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_black_alpha(150))
                    .corner_radius(4.0)
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        // clear of the Volume / Isosurface switch (bottom left) and the toasts
                        ui.set_max_width((view.width() - 2.0 * 280.0).max(260.0));
                        ui.add(egui::Label::new(RichText::new(label).color(Color32::from_gray(210))).wrap());
                    });
            });
        bottom = r.response.rect.top() - 8.0;
    }
    if let Some(top) = sampling_card(ctx, view, bottom, input, acts) {
        bottom = top - 8.0;
    }
    if let Some(top) = relax_card(ctx, view, bottom, input, acts) {
        bottom = top - 8.0;
    }
    toast(ctx, view, bottom, st);
    view_switch(ctx, view, st, input, acts);
    legend(ctx, view, st, input);
    draft_banner(ctx, view, chip, input);
    // Q3 of Appendix B (draft mode's per-atom uncertainty tint) is the sphere colour itself:
    // `gfx::risk_tint` of `viewer::Viewer::atom_risk`.
}

/// Draft mode's red "extrapolating" banner (Q2 of Appendix B): the model is answering beyond what
/// it has learned. It stacks directly under the status chip (same left edge, top-left of the view)
/// so it never covers the chip's time, step and temperature.
fn draft_banner(ctx: &egui::Context, view: Rect, chip: Rect, input: &PanelInput<'_>) {
    let Some(text) = &input.draft_banner else { return };
    egui::Area::new(Id::new("draft-banner"))
        .pivot(Align2::LEFT_TOP)
        .fixed_pos(pos2(chip.left(), chip.bottom() + 8.0))
        .interactable(false)
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgba_unmultiplied(150, 20, 20, 220))
                .stroke(Stroke::new(1.0, RED))
                .corner_radius(6.0)
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.set_max_width((view.width() - 120.0).clamp(240.0, 560.0));
                    ui.label(RichText::new(text).color(Color32::WHITE).strong());
                });
        });
}

/// The state line of the old panel (row 2: every state keeps its text).
pub(super) fn state(input: &PanelInput<'_>) -> (bool, String) {
    let snap = input.snapshot;
    let sampling = snap.and_then(|s| s.dynamics.sampling.as_ref().filter(|_| crate::sim::sampling_active(&s.dynamics)));
    if let Some(p) = sampling {
        (true, format!("sampling the teacher: {} {}/{}", crate::ui::activity::words::pass_stage(p), p.done, p.total))
    } else if input.scene.nuclei.is_empty() && !input.busy {
        // playing or not: nothing moves until an atom is placed
        (false, super::EMPTY_STATUS.into())
    } else if input.playing {
        (true, "playing".into())
    } else if input.busy {
        (true, "solving…".into())
    } else if let Some(s) = snap {
        if s.dynamics.forces_stale {
            (false, "paused — not evaluated (see the error)".into())
        } else if s.scene == *input.scene {
            (false, "paused — converged".into())
        } else {
            (false, "paused — density is from an earlier geometry".into())
        }
    } else {
        (false, "waiting for the first solve".into())
    }
}

fn sep(ui: &mut egui::Ui) {
    ui.add(egui::Separator::default().vertical().spacing(6.0));
}

/// The chip's lines under the state: the bond references being solved in the background (dim,
/// M6 Task 3 fix round 2 — background work, never "solving…"), the status line (amber while it
/// asks for something, like "paused: … press Play to continue", dim otherwise) until it is
/// replaced or cleared, and the driver's refusal (amber) until the next accepted command — in
/// place, as the old panel kept them.
pub fn chip_notes(st: &UiState, input: &PanelInput<'_>) -> Vec<(Color32, String)> {
    let mut v = Vec::new();
    if st.volume_mode == super::VOLUME_DOTS {
        v.push((theme::INK_DIM, super::dock::orbitals::dots_note(st)));
    }
    if let Some(r) = input.snapshot.and_then(|s| s.dynamics.bond_reference) {
        v.push((theme::INK_DIM, bond_reference_line(&r)));
    }
    if let Some(g) = input.pseudo_generating {
        v.push((theme::INK_DIM, generating_line(g)));
    }
    if !st.status.is_empty() {
        let asks = st.status.starts_with("paused:") || st.status.contains("press ");
        v.push((if asks { super::AMBER } else { theme::INK_DIM }, st.status.clone()));
    }
    if let Some(r) = input.snapshot.and_then(|s| s.dynamics.last_rejection.as_ref()) {
        v.push((super::AMBER, format!("refused: {r}")));
    }
    if let Some(p) = teach_prompt(input) {
        v.push((super::AMBER, format!("stuck on DFT: {p}")));
    }
    v
}

/// The teach-around prompt (`sim::TeachOffer::prompt`): consecutive class-rule fallbacks blocked by
/// a sample far from here; `None` while a pass runs.
pub fn teach_prompt<'a>(input: &'a PanelInput<'_>) -> Option<&'a str> {
    input.model().and_then(|m| m.teach_offer.as_ref()).and_then(|o| o.prompt.as_deref()).filter(|_| !input.sampling())
}

/// "generating the H pseudopotential (≈ 1 min: its dimer check)… 12 s" — the first solve of a scene waits for its tables
/// (M7 Task 11; controller's cost condition: the minute is said, never silent). Without the gate: "(seconds)".
pub fn generating_line((z, secs, gated): (u8, f64, bool)) -> String {
    let what = if gated { "≈ 1 min: its dimer check" } else { "seconds" };
    format!("generating the {} pseudopotential ({what})… {secs:.0} s", units::isotopes::symbol(z))
}

/// "bond references: O 2/3… (14 s)": the species, the multiplicity being solved of the scan's
/// total, and the time spent (the old panel's status line, M6 Task 3).
pub fn bond_reference_line(r: &sim::ReferenceProgress) -> String {
    format!(
        "bond references: {} {}/{}… ({:.0} s)",
        units::isotopes::symbol(r.z),
        (r.done + 1).min(r.total),
        r.total,
        r.elapsed_s
    )
}

/// Draws the chip and returns its rect (the draft banner stacks under it).
fn status_chip(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>) -> Rect {
    let notes = chip_notes(st, input);
    let errors: Vec<String> = input
        .last_error
        .map(str::to_string)
        .into_iter()
        .chain(input.snapshot.and_then(|s| s.dynamics.last_error.as_ref()).map(|e| format!("paused by a solver failure: {e}")))
        .collect();
    // the activity headline names what runs (spec §3.1); idle states keep their old texts. An
    // empty headline text is background work with no state of its own (bond references: the dim
    // note line below), so the old state text stays.
    let headline = st.activity.headline.peek().clone();
    let (busy, text) = match headline.as_ref().filter(|h| !h.text.is_empty()) {
        Some(h) => (true, h.text.clone()),
        None => state(input),
    };
    if headline.is_some() {
        ctx.request_repaint_after(activity::headline::REFRESH);
    }
    egui::Area::new(Id::new("status-chip"))
        .fixed_pos(view.min + Vec2::splat(12.0))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            let mut frame = theme::card();
            if !errors.is_empty() {
                frame = frame.stroke(Stroke::new(1.0, RED)).fill(Color32::from_rgba_premultiplied(70, 14, 20, 225));
            }
            frame.show(ui, |ui| {
                ui.set_max_width((view.width() - 300.0).clamp(260.0, 640.0));
                ui.horizontal(|ui| {
                    if busy {
                        ui.add(egui::Spinner::new().size(12.0).color(theme::ACCENT));
                    } else {
                        theme::light(ui, if errors.is_empty() { theme::INK_FAINT } else { RED });
                    }
                    ui.label(RichText::new(text).strong());
                    if let Some(a) = headline.as_ref().and_then(|h| h.audit.as_ref()) {
                        theme::pill(ui, a, GREEN).on_hover_text("a routine check of the model against DFT runs on the background teacher; the model keeps answering meanwhile");
                    }
                    if let Some(o) = headline.as_ref().and_then(|h| h.orbitals.as_ref()) {
                        theme::pill(ui, o, theme::ACCENT).on_hover_text("the orbital view's display eigensolve (Orbitals tab); it shows when it lands");
                    }
                    if let Some(s) = input.snapshot {
                        let d = &s.dynamics;
                        if s.backend == Backend::Estimated {
                            theme::pill(ui, source_label(d.source), source_colour(d.source)).on_hover_text(super::SOURCE_TIP);
                            if d.history_guess.last() == Some(&true) {
                                theme::pill(ui, "draft guess", RED).on_hover_text("draft mode: this step is the model's guess, also where it would have fallen back to DFT");
                            }
                        } else {
                            theme::pill(ui, &format!("DFT · {}", s.engine.label()), SPARK_E);
                        }
                        sep(ui);
                        ui.label(RichText::new(format!("t {:.2} fs", d.sim_time_fs())).monospace());
                        ui.label(RichText::new(format!("step {}", d.step)).monospace().color(theme::INK_DIM));
                        sep(ui);
                        ui.label(RichText::new(format!("T {}", super::averaged_temperature_text(d, &st.temp_avg, 0, st.temp_units))).monospace())
                            .on_hover_text(format!("kinetic temperature averaged over the last second; this step: {}", super::temperature_text(d, 1, st.temp_units)));
                        if let Some((t, tau)) = d.thermostat {
                            ui.label(RichText::new(format!("CSVR {}", st.temp_units.show(t, 0))).color(theme::INK_DIM)).on_hover_text(format!("thermostat on: target {t} K, τ {tau} fs (Physics menu)"));
                        }
                    }
                });
                for e in &errors {
                    ui.add(egui::Label::new(RichText::new(e).color(Color32::from_rgb(255, 210, 210))).wrap());
                }
                for (c, n) in &notes {
                    ui.add(egui::Label::new(RichText::new(n).size(11.5).color(*c)).wrap());
                }
            });
        })
        .response
        .rect
}

/// The title, the Apply button and the Cancel button of [`model_offer_card`] (the tests and the docs quote them).
pub const OFFER_TITLE: &str = "This model was taught at other settings than this scene's";
pub const OFFER_APPLY: &str = "Apply the model's settings and load";
pub const OFFER_CANCEL: &str = "Cancel";
pub const OFFER_NOTE: &str = "Apply changes only the settings listed (the nuclei stay where they are), then loads the model. Cancel leaves everything as it is.";

/// The card for a picked model file whose settings differ from the scene's (`ui::model_offer`): each
/// differing setting as "model: X · scene: Y", what cannot be applied and why, and the two buttons.
fn model_offer_card(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let Some(o) = st.model_offer.as_ref() else { return };
    let sampling = input.sampling();
    let blocked = o.blocked();
    egui::Area::new(Id::new("model-offer-card"))
        .pivot(Align2::CENTER_CENTER)
        .fixed_pos(view.center())
        .order(Order::Foreground)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().stroke(Stroke::new(1.0, super::AMBER.gamma_multiply(0.6))).show(ui, |ui| {
                ui.set_width(460.0f32.min(view.width() - 40.0));
                ui.label(RichText::new(OFFER_TITLE).strong());
                ui.label(RichText::new(o.path.display().to_string()).monospace().size(11.0).color(theme::INK_DIM));
                ui.add_space(4.0);
                for r in &o.rows {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(r.name).strong());
                        ui.label(RichText::new(r.versus()).monospace().size(11.5));
                    });
                    if let Some(b) = &r.blocked {
                        ui.add(egui::Label::new(RichText::new(format!("cannot be applied: {b}")).size(11.0).color(super::AMBER)).wrap());
                    }
                    if let Some(n) = &r.note {
                        ui.add(egui::Label::new(RichText::new(n).size(11.0).color(theme::INK_DIM)).wrap());
                    }
                }
                ui.add_space(4.0);
                theme::note(ui, OFFER_NOTE);
                ui.horizontal(|ui| {
                    let apply = egui::Button::new(RichText::new(OFFER_APPLY).strong());
                    let why = if sampling { Some("a sampling pass is running: cancel it first") } else { blocked.map(|_| "a setting listed above cannot be applied") };
                    let r = ui.add_enabled(why.is_none(), apply);
                    let r = match why {
                        Some(w) => r.on_disabled_hover_text(w),
                        None => r.on_hover_text("set the grid, σ, charge, multiplicity, ionic potential and isotopes listed to the model's, then load it"),
                    };
                    if r.clicked() {
                        acts.push(Action::ApplyModelSettings);
                    }
                    if ui.button(OFFER_CANCEL).on_hover_text("close this card; nothing is changed or loaded").clicked() {
                        acts.push(Action::CancelModelOffer);
                    }
                });
            });
        });
}

/// Returns the card's top edge when it is shown.
fn sampling_card(ctx: &egui::Context, view: Rect, bottom: f32, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Option<f32> {
    let p = input.snapshot.and_then(|s| s.dynamics.sampling.as_ref()).filter(|_| input.sampling())?;
    let r = egui::Area::new(Id::new("sampling-card"))
        .pivot(Align2::CENTER_BOTTOM)
        .fixed_pos(pos2(view.center().x, bottom))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().show(ui, |ui| {
                ui.set_width(520.0f32.min(view.width() - 40.0));
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Sampling pass").strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Cancel").on_hover_text("end the pass now: what it sampled is kept; it is refitted only once there are enough samples to fit (Q′, the model's number of directions), else the Log says so and Sample more adds to them").clicked() {
                            acts.push(Action::CancelSampling);
                        }
                    });
                });
                dock::progress(ui, p, input.model());
                theme::note(ui, "Stages: isolated-atom references, perturbations, pair scans, hot MD burst.");
            });
        });
    Some(r.response.rect.top())
}

/// The running relax's card (bottom centre): its readout and Stop. Returns the card's top edge.
fn relax_card(ctx: &egui::Context, view: Rect, bottom: f32, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Option<f32> {
    let r = input.snapshot.and_then(|s| s.dynamics.relax.as_ref()).filter(|r| r.converged.is_none())?;
    let resp = egui::Area::new(Id::new("relax-card"))
        .pivot(Align2::CENTER_BOTTOM)
        .fixed_pos(pos2(view.center().x, bottom))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().show(ui, |ui| {
                ui.set_width(520.0f32.min(view.width() - 40.0));
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Relaxing (FIRE)").strong()).on_hover_text(super::RELAX_TIP);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Stop").on_hover_text(super::STOP_RELAX_TIP).clicked() {
                            acts.push(Action::StopRelax);
                        }
                    });
                });
                for l in super::relax_lines(r) {
                    ui.add(egui::Label::new(RichText::new(l).monospace().size(11.5)).wrap());
                }
                theme::note(ui, "FIRE steps are not time: t and the step count stay put; the energy drained is booked as work done by tools.");
            });
        });
    Some(resp.response.rect.top())
}

/// A status line toasts at the bottom right of the view; a refusal above the toolbar (bottom centre,
/// over the density label and the sampling card).
fn toast(ctx: &egui::Context, view: Rect, bottom: f32, st: &mut UiState) {
    let Some((kind, text)) = st.log.current_toast() else { return };
    let text = text.to_string();
    let mut open_log = false;
    let refused = kind == super::logbook::LogKind::Refused;
    let (pivot, at) = if refused {
        (Align2::CENTER_BOTTOM, pos2(view.center().x, bottom))
    } else {
        // above the estimated view's "Bonds now" button
        let lift = if st.backend == crate::sim::Backend::Estimated { 52.0 } else { 0.0 };
        (Align2::RIGHT_BOTTOM, view.max - Vec2::new(14.0, 14.0 + lift))
    };
    egui::Area::new(Id::new("toast"))
        .pivot(pivot)
        .fixed_pos(at)
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().show(ui, |ui| {
                ui.set_max_width(420.0);
                ui.horizontal(|ui| {
                    theme::light(ui, kind.colour());
                    let t = RichText::new(&text);
                    let t = if kind == super::logbook::LogKind::Refused { t.color(super::AMBER) } else { t };
                    ui.add(egui::Label::new(t).wrap());
                    if ui.add(egui::Button::new(RichText::new("· Log").color(theme::INK_FAINT)).frame(false)).clicked() {
                        open_log = true;
                    }
                });
            });
        });
    if open_log {
        st.open_tab(super::DockTab::Log);
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(500));
}

/// Bottom-left: the axes gizmo (with the axes toggle) and the quick Volume / Isosurface switch.
fn view_switch(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let dots_tip = format!("random samples of the field drawn, resampled every frame (View › Density › shimmer). {}", super::dock::orbitals::dots_note(st));
    egui::Area::new(Id::new("view-switch"))
        .pivot(Align2::LEFT_BOTTOM)
        .fixed_pos(pos2(view.left() + 14.0, view.bottom() - 14.0))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if st.show_axes {
                    gizmo(ui, input);
                }
                egui::Frame::new().fill(theme::CARD).corner_radius(7).show(ui, |ui| {
                    if let Some(m) = theme::segmented(
                        ui,
                        st.volume_mode,
                        &[
                            (0, "Volume", "density (or an orbital, or the spin density) as a glowing volume (View › Density for gamma and opacity)"),
                            (1, "Isosurface", "density as a surface at iso × ρ max; an orbital or the spin density as a ± isosurface (View › Density, and the iso slider in the Orbitals tab)"),
                            (2, "Dots", dots_tip.as_str()),
                        ],
                    ) {
                        acts.push(Action::VolumeMode(m));
                    }
                });
            });
        });
}

/// What the legend chip shows (orbital-viewer spec §3): the colour pair or the palette in use.
pub struct Legend {
    /// Swatches (linear display colour, name) and the line under them.
    pub swatches: Vec<([f32; 3], String)>,
    pub text: String,
    /// The tooltip: the facts of spec §2 that apply to what is drawn.
    pub tip: String,
}

/// The legend of the orbital layer for the panel's `show` mode and the payload on screen (`None`
/// for the density, or with nothing of that mode to draw yet): phase ±, spin ↑/↓, or the palette
/// slots of the included orbitals with the scale in use.
pub fn legend_of(st: &UiState, input: &PanelInput<'_>) -> Option<Legend> {
    use super::dock::orbitals as ot;
    use super::ShowMode;
    let o = input.snapshot?.dynamics.orbitals.as_ref().filter(|o| o.field.is_some())?;
    let signed = matches!(o.field, Some(sim::OrbitalField::Gpu { signed: true, .. } | sim::OrbitalField::Cpu { signed: true, .. }));
    Some(match (st.show, &o.what) {
        (ShowMode::Orbitals, sim::OrbitalRequest::Orbital { .. }) if signed => Legend {
            swatches: vec![(render::volume::PHASE_POS, "+".into()), (render::volume::PHASE_NEG, "−".into())],
            text: if st.volume_mode == super::VOLUME_ISO {
                format!("sign of ψ̃ · iso ±{:.2} of max|ψ| · only the relative sign of its lobes means anything", st.phase_iso)
            } else {
                "sign of ψ̃ · only the relative sign of its lobes means anything".into()
            },
            tip: format!("{} {}.", ot::KS_NOTE, ot::SIGN_TIP),
        },
        (ShowMode::Orbitals, sim::OrbitalRequest::Orbital { .. }) => Legend {
            swatches: Vec::new(),
            text: "ψ̃²: where an electron in this orbital is likely to be (the density's colour ramp)".into(),
            tip: ot::KS_NOTE.into(),
        },
        (ShowMode::Spin, sim::OrbitalRequest::SpinDensity) => Legend {
            swatches: vec![(render::volume::SPIN_UP, "↑ excess".into()), (render::volume::SPIN_DOWN, "↓ excess".into())],
            text: if st.volume_mode == super::VOLUME_ISO { format!("ρ↑ − ρ↓ · iso ±{:.2} of its scale", st.spin_iso) } else { "ρ↑ − ρ↓".into() },
            tip: format!("{} {}", ot::SPIN_NOTE, ot::KS_NOTE),
        },
        (ShowMode::Coloured, sim::OrbitalRequest::AllColoured { include, physical, .. }) => {
            let mut swatches: Vec<([f32; 3], String)> = Vec::new();
            let mut slots = Vec::new();
            for r in o.rows.iter().filter(|r| match include {
                Some(tracks) => tracks.contains(&r.track),
                None => r.occupation > 0.0,
            }) {
                if slots.contains(&r.slot) {
                    continue; // a ↓ orbital shares its ↑ partner's colour
                }
                slots.push(r.slot);
                let c = providers::orbitals::ORBITAL_PALETTE[r.slot as usize % providers::orbitals::ORBITAL_PALETTE.len()];
                swatches.push((c, r.label.to_string()));
            }
            let scale = if *physical { "physical scale: hue = which orbital dominates, opacity = the density" } else { "each orbital scaled to its own peak" };
            Legend { swatches, text: scale.into(), tip: format!("{} Slots repeat beyond 12 orbitals.", ot::KS_NOTE) }
        }
        _ => return None,
    })
}

/// The legend chip over the 3D view's bottom left, above the Volume / Isosurface switch.
fn legend(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>) {
    let Some(l) = legend_of(st, input) else { return };
    egui::Area::new(Id::new("orbital-legend"))
        .pivot(Align2::LEFT_BOTTOM)
        .fixed_pos(pos2(view.left() + 14.0, view.bottom() - 70.0))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card()
                .show(ui, |ui| {
                    ui.set_max_width((view.width() * 0.45).clamp(200.0, 520.0));
                    ui.horizontal_wrapped(|ui| {
                        for (c, name) in &l.swatches {
                            let (rect, _) = ui.allocate_exact_size(Vec2::splat(11.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 3.0, dock::orbitals::swatch(*c));
                            ui.label(RichText::new(name).size(11.5));
                        }
                    });
                    ui.add(egui::Label::new(RichText::new(&l.text).size(11.0).color(theme::INK_DIM)).wrap());
                })
                .response
                .on_hover_text(&l.tip);
        });
}

/// The camera's x (red), y (green), z (blue) axes, seen from the eye.
fn gizmo(ui: &mut egui::Ui, input: &PanelInput<'_>) {
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(46.0), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    p.circle(c, 21.0, Color32::from_rgba_premultiplied(18, 17, 24, 200), Stroke::new(1.0, theme::LINE));
    let view = input.camera.view();
    let mut axes = [
        (glam::Vec3::X, Color32::from_rgb(232, 96, 90), "x"),
        (glam::Vec3::Y, Color32::from_rgb(120, 210, 126), "y"),
        (glam::Vec3::Z, Color32::from_rgb(110, 168, 255), "z"),
    ]
    .map(|(a, col, l)| (view.transform_vector3(a), col, l));
    // far axes first
    axes.sort_by(|a, b| a.0.z.total_cmp(&b.0.z));
    for (v, col, l) in axes {
        let tip: Pos2 = c + Vec2::new(v.x, -v.y) * 15.0;
        p.line_segment([c, tip], Stroke::new(2.0, col));
        p.text(c + Vec2::new(v.x, -v.y) * 19.0, Align2::CENTER_CENTER, l, egui::FontId::monospace(8.5), col);
    }
    r.on_hover_text("the world axes as the camera sees them: x red, y green (up), z blue (View › Display aids › axes)");
}

/// The estimated view's "Bonds now" (bottom right; user request 2026-10-19): the surrogate's steps
/// carry no density, so the bonds and angles are an earlier DFT step's; this asks for one teacher
/// solve between steps (`Action::RefreshBonds`) and shows the bonds, lengths and angles. It spins
/// until the graph is fresh (at most 20 s).
fn bonds_now(ctx: &egui::Context, view: Rect, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if st.backend != crate::sim::Backend::Estimated {
        return;
    }
    let Some(s) = input.snapshot else { return };
    if s.scene.nuclei.len() < 2 {
        return;
    }
    let stale = s.dynamics.bonds_stale || s.dynamics.bonds.is_none();
    let busy = st.bonds_refresh.is_some_and(|t| t.elapsed().as_secs_f64() < 20.0) && (stale || st.bonds_refresh.is_some_and(|t| t.elapsed().as_secs_f64() < 0.6));
    if !busy {
        st.bonds_refresh = None;
    }
    egui::Area::new(Id::new("bonds-now"))
        .pivot(Align2::RIGHT_BOTTOM)
        .fixed_pos(view.max - Vec2::splat(14.0))
        .order(Order::Middle)
        .show(ctx, |ui| {
            // the bond references come first (an isolated-atom solve per species): the graph
            // cannot be read without them, and a press waits for them in the driver
            let pending = &s.dynamics.bonds_pending;
            if !pending.is_empty() {
                let z = s.dynamics.bond_reference.as_ref().map_or(pending[0], |r| r.z);
                let pct = s.dynamics.bond_reference.as_ref().filter(|r| r.total > 0).map_or(String::new(), |r| format!(" {:.0} %", 100.0 * r.done as f64 / r.total as f64));
                egui::Frame::new()
                    .fill(theme::PANEL_3)
                    .stroke(egui::Stroke::new(1.0, theme::LINE))
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(14, 9))
                    .show(ui, |ui| {
                        ui.set_min_width(172.0);
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new().size(16.0).color(theme::INK));
                            ui.label(egui::RichText::new(format!("Bond references: {}{pct}", units::elements::symbol(z))).strong().size(14.0).color(theme::INK));
                        });
                    })
                    .response
                    .on_hover_text("the bonds are read against isolated-atom references, one per element, solved in the background first; a press made now runs once they are done");
                return;
            }
            if busy {
                // a spinner beside the word (not over it), in the button's place
                egui::Frame::new()
                    .fill(theme::GLOW)
                    .stroke(egui::Stroke::new(1.0, theme::ACCENT))
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(14, 9))
                    .show(ui, |ui| {
                        ui.set_min_width(172.0);
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new().size(16.0).color(egui::Color32::WHITE));
                            ui.label(egui::RichText::new("Solving…").strong().size(14.0).color(egui::Color32::WHITE));
                        });
                    })
                    .response
                    .on_hover_text("one DFT solve at the current positions, then the bonds and angles");
                return;
            }
            let word = if stale { "Bonds & angles now" } else { "Bonds & angles: current" };
            let b = egui::Button::new(egui::RichText::new(word).strong().size(14.0).color(egui::Color32::WHITE))
                .fill(if stale { theme::GLOW } else { theme::PANEL_3 })
                .stroke(egui::Stroke::new(1.0, theme::ACCENT))
                .corner_radius(8)
                .min_size(Vec2::new(200.0, 36.0));
            let r = ui.add_enabled(!input.sampling(), b).on_hover_text(
                "the model's steps carry no density, so the bonds and angles shown are from the last DFT step: this solves the density once at the current positions (between two steps; a running simulation carries on) and shows the bonds, their lengths and the angles",
            );
            if r.clicked() {
                acts.push(Action::RefreshBonds);
            }
        });
}
