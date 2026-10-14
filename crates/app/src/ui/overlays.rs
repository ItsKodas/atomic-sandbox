//! What the chrome draws over the 3D view (the rect the panels leave): the status chip
//! (top-left: state, who answered, time, step, T, thermostat; red with the error when there is
//! one), the atom card (top-right), the long first-pass offer (top-right, under the card), the
//! sampling card (bottom centre, while a pass runs), the faded-density label (bottom centre),
//! toasts (bottom-right) and the Volume / Isosurface switch with the axes gizmo (bottom-left).
//! The distance guides, drag spring and pin squares stay with the viewer (`viewer.rs`), which
//! projects them from the camera; the cell, floor, axes and placement plane are 3D lines
//! (`overlay.rs`).
use super::dock;
use super::{source_colour, source_label, theme, Action, PanelInput, UiState, RED, SPARK_E};
use crate::sim::Backend;
use egui::{pos2, Align2, Color32, Id, Order, Pos2, Rect, RichText, Stroke, Vec2};

pub fn show(ctx: &egui::Context, view: Rect, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if view.width() < 50.0 || view.height() < 50.0 {
        return;
    }
    let chip = status_chip(ctx, view, st, input);
    let mut right_y = view.top() + 12.0;
    if st.show_atom_card {
        if let Some((i, n)) = input.selection() {
            let r = crate::atom_card::show(ctx, pos2(view.right() - 12.0, right_y), n, i);
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
    let mut bottom = view.bottom() - 14.0;
    if let Some(label) = &input.density_label {
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
    toast(ctx, view, bottom, st);
    view_switch(ctx, view, st, input, acts);
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
        (true, format!("sampling the teacher: {} {}/{}", p.stage, p.done, p.total))
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

/// The chip's lines under the state: the status line (amber while it asks for something, like
/// "paused: … press Play to continue", dim otherwise) until it is replaced or cleared, and the
/// driver's refusal (amber) until the next accepted command — in place, as the old panel kept them.
pub fn chip_notes(st: &UiState, input: &PanelInput<'_>) -> Vec<(Color32, String)> {
    let mut v = Vec::new();
    if !st.status.is_empty() {
        let asks = st.status.starts_with("paused:") || st.status.contains("press ");
        v.push((if asks { super::AMBER } else { theme::INK_DIM }, st.status.clone()));
    }
    if let Some(r) = input.snapshot.and_then(|s| s.dynamics.last_rejection.as_ref()) {
        v.push((super::AMBER, format!("refused: {r}")));
    }
    v
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
    let (busy, text) = state(input);
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
                        ui.label(RichText::new(format!("T {}", super::temperature_text(d, 0, st.temp_units))).monospace());
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
                        if ui.button("Cancel").on_hover_text("end the pass now: what it sampled is kept and refitted").clicked() {
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
        (Align2::RIGHT_BOTTOM, view.max - Vec2::splat(14.0))
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
        st.dock_open = true;
        st.dock_tab = super::DockTab::Log;
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(500));
}

/// Bottom-left: the axes gizmo (with the axes toggle) and the quick Volume / Isosurface switch.
fn view_switch(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
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
                        &[(0, "Volume", "density as a glowing volume (View › Density for gamma and opacity)"), (1, "Isosurface", "density as a surface at iso × ρ max (View › Density)")],
                    ) {
                        acts.push(Action::VolumeMode(m));
                    }
                });
            });
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
