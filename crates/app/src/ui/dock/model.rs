//! The Model tab (the estimated solver, spec §6; rows 10–45 of the mapping): the honesty line,
//! the model's state per snapshot (who answered, uncertainty against its bound, trust, samples,
//! epoch, fits, fallbacks, teacher gap, last audit, mismatch work), the warning slots, the
//! sampling pass (progress, last pass, the long first-pass offer), the audit cadence, the
//! sampling plan, Start / Sample more / Cancel, and the model file.
use super::super::actions::{expire_arming, forget_armed, sample_button};
use super::super::{
    duration_label, search, source_colour, source_label, theme, trust, Action, PanelInput, UiState, AMBER, GREEN, RED, SOURCE_TIP,
};
use crate::scene::model_path;
use crate::sim::{Backend, ModelStatus};
use egui::{Color32, RichText, Ui};
use sim::SamplingPlan;

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    expire_arming(st);
    ui.horizontal_wrapped(|ui| {
        theme::note(ui, "estimated = a model fitted to this scene's DFT; valid only where it has been taught — it falls back to DFT when unsure");
        theme::info(ui, "The estimated solver is a model of this DFT, not of nature: it inherits every approximation of its teacher (smearing, LSDA, the box) and adds its own fit error. Help › Physics, honestly has the long form.");
    });
    if st.backend != Backend::Estimated {
        theme::note(ui, "The estimated solver is not running: pick Estimated in the toolbar's Solver group. Its model, sampling pass and audits show here.");
        return;
    }
    let snap = input.snapshot.filter(|s| s.backend == Backend::Estimated);
    let d = snap.map(|s| &s.dynamics);
    let m: Option<&ModelStatus> = input.model();
    let sampling = input.sampling();
    super::group(ui, st, "d-model", |ui, _| {
        if let (Some(d), Some(m)) = (d, m) {
            readouts(ui, d, m);
        }
    });
    if let Some(m) = m {
        if let Some(why) = &m.provenance_mismatch {
            warnbox(ui, &format!("{why}: every step falls back to DFT and teaches nothing — restore those settings, or forget the model and sample this scene"));
        }
        if let Some(r) = &m.autoload_refused {
            warnbox(ui, &format!("this session's model was not restored: {r}"));
            if ui.button(RichText::new("Forget it and sample this scene").color(AMBER)).clicked() {
                acts.push(Action::SampleAfterRefusal);
            }
        }
    }
    // the sampling pass
    if let Some(p) = d.and_then(|d| d.sampling.as_ref()) {
        if sampling {
            progress(ui, p, m);
        } else {
            theme::note(
                ui,
                format!(
                    "last pass {}: {}/{} jobs, skipped {}, failed {}",
                    if p.cancelled { "cancelled" } else { "finished" },
                    p.done,
                    p.total,
                    p.skipped,
                    p.failed
                ),
            );
        }
    }
    if let Some(o) = m.and_then(|m| m.pass_offer.as_ref()).filter(|_| !sampling) {
        offer(ui, o, acts);
    }
    super::group(ui, st, "d-audits", |ui, st| {
        theme::section(ui, "Audits");
        ui.horizontal(|ui| {
            ui.label("DFT audit every");
            let mut n = st.audit_every;
            if ui
                .add(egui::DragValue::new(&mut n).range(0..=1_000_000).speed(10.0))
                .on_hover_text("every n-th step the model answers, the DFT teacher checks it at that geometry and the gap is recorded (a failed audit untrusts the model until a refit). 2000 by default. With a GPU teacher the audit runs on a second teacher in the background while the model keeps stepping (ruling S14), so the cadence costs no speed: the check simply arrives a few hundred steps later. With the CPU teacher it runs in the step and sets the speed. 0 = never (unsafe: nothing checks the model while it answers).")
                .changed()
            {
                acts.push(Action::SetAudit(n));
            }
            ui.label("answered steps");
            // what the worker runs (after a panic recovery it re-applies this value)
            if let Some(m) = m.filter(|m| m.audit_every != st.audit_every) {
                ui.colored_label(AMBER, format!("(solver: {})", m.audit_every));
            }
        });
        if st.audit_every == 0 {
            ui.colored_label(AMBER, "audits off: nothing checks the model against DFT while it answers (unsafe)");
        }
    });
    let fitted = m.is_some_and(|m| m.fitted);
    let r = egui::CollapsingHeader::new("Sampling plan")
        .id_salt("sampling-plan")
        .default_open(!fitted)
        .open(search::opens(st, "d-plan"))
        .show(ui, |ui| plan_fields(ui, &mut st.plan));
    search::mark(ui, st, "d-plan", r.header_response.rect);
    super::group(ui, st, "d-sample", |ui, st| sample_row(ui, st, input, acts));
    super::group(ui, st, "d-model-file", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Model file");
        });
        model_file_buttons(ui, st, input, acts);
        ui.label(RichText::new(model_path(&st.scene_path).display().to_string()).monospace().size(11.0).color(theme::INK_DIM));
    });
    let r = egui::CollapsingHeader::new("What this model is (and is not)")
        .id_salt("model-about")
        .default_open(false)
        .open(search::opens(st, "d-model-about"))
        .show(ui, |ui| {
            theme::note(ui, "A MODEL uncertainty: the surrogate inherits every DFT approximation (smearing, LSDA) on top of it. An untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust.");
            theme::note(ui, "Box-affected teacher answers (density on the outermost grid plane above 2e-5 × (h / 0.25 Bohr)² e/Bohr³) are used for that step but never trained on.");
        });
    search::mark(ui, st, "d-model-about", r.header_response.rect);
    // SEAM(Q1, m5b-viewer-extras): the draft-mode switch also lands here (Appendix B).
    theme::queued_row(ui, "Draft mode (unsafe)");
}

fn warnbox(ui: &mut Ui, text: &str) {
    egui::Frame::new()
        .fill(AMBER.gamma_multiply(0.07))
        .stroke(egui::Stroke::new(1.0, AMBER.gamma_multiply(0.45)))
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(9, 7))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.5).color(AMBER));
        });
}

/// The running pass's progress bar and counts (also on the sampling card in the 3D view).
pub fn progress(ui: &mut Ui, p: &sim::SamplingProgress, m: Option<&ModelStatus>) {
    let frac = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
    let eta = m.and_then(|m| m.pass_eta_s).map_or(String::new(), |t| format!(" — about {} left", duration_label(t)));
    ui.add(egui::ProgressBar::new(frac).show_percentage().text(format!("{} — {}/{}{eta}", p.stage, p.done, p.total)));
    let e = if p.last_energy.is_finite() { format!("{:+.6} Ha", p.last_energy) } else { "—".into() };
    ui.weak(format!(
        "last teacher energy {e} · skipped {} · failed {} — the dynamics are paused; edits are allowed",
        p.skipped, p.failed
    ));
}

/// The long first-pass offer and its "Start anyway" (also on the offer card in the 3D view).
pub fn offer(ui: &mut Ui, o: &crate::sim::PassOffer, acts: &mut Vec<Action>) {
    ui.colored_label(
        AMBER,
        format!(
            "a first sampling pass here takes about {} ({} solves × ≈ {:.1} s; the panel's plan, {} solves, ≈ {}) — that is already the fewest solves that can earn trust on this scene. A smaller grid is faster.",
            duration_label(o.eta_s),
            o.solves,
            o.per_solve_s,
            o.full_solves,
            duration_label(o.full_eta_s)
        ),
    );
    if ui.button(RichText::new(format!("Start anyway (≈ {})", duration_label(o.eta_s))).color(AMBER)).clicked() {
        acts.push(Action::StartOffered(o.plan.clone()));
    }
}

/// Start sampling / Sample more (armed "Start anyway" above 2 min), the plan's cost, Cancel.
fn sample_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let sampling = input.sampling();
    let (label, long, cost) = sample_button(st, input);
    ui.horizontal_wrapped(|ui| {
        let armed = label.starts_with("Start anyway");
        let text = if armed { RichText::new(label).color(AMBER) } else { RichText::new(label).strong() };
        let b = if armed { egui::Button::new(text) } else { egui::Button::new(text.color(Color32::from_rgb(0x17, 0x11, 0x2a))).fill(theme::ACCENT) };
        if ui
            .add_enabled(!sampling, b)
            .on_hover_text("one pass over the current scene with the plan above: isolated-atom references, random perturbations, pair scans, a hot MD burst; the samples are added to the model's and it is refitted. Pauses the dynamics. A pass longer than 2 min asks for a second click.")
            .clicked()
        {
            acts.push(Action::Sample { long });
        }
        if let Some(c) = cost {
            ui.weak(c);
        }
        if ui
            .add_enabled(sampling, egui::Button::new("Cancel"))
            .on_hover_text("end the pass now: what it sampled is kept and refitted")
            .clicked()
        {
            acts.push(Action::CancelSampling);
        }
    });
}

/// Save model / Load model / Forget model (two clicks within 4 s): File ▸ Estimated model and
/// the Model tab.
pub fn model_file_buttons(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    expire_arming(st);
    let estimated = st.backend == Backend::Estimated;
    let m = input.model();
    let sampling = input.sampling();
    let fitted = m.is_some_and(|m| m.fitted);
    let path = model_path(&st.scene_path);
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(estimated && fitted, egui::Button::new("Save model"))
            .on_hover_text(format!("write the model, its fit and its samples to {} (next to the scene file)", path.display()))
            .clicked()
        {
            acts.push(Action::SaveModel);
        }
        if ui
            .add_enabled(estimated && !sampling, egui::Button::new("Load model"))
            .on_hover_text(format!(
                "read {} — refused unless it was trained on this scene's grid, σ, charge, multiplicity and teacher",
                path.display()
            ))
            .clicked()
        {
            acts.push(Action::LoadModel);
        }
        let armed = forget_armed(st);
        let samples = m.map_or(0, |m| m.samples);
        let forget = if armed {
            egui::Button::new(RichText::new(format!("Click again to forget {samples} samples")).color(RED))
        } else {
            egui::Button::new(RichText::new("Forget model…").color(theme::DANGER_INK))
        };
        if ui
            .add_enabled(estimated && (m.is_some_and(|m| m.samples > 0 || m.fitted) || sampling), forget)
            .on_hover_text("drop the model, its samples and references (two clicks within 4 s); the teacher answers every step until the next pass")
            .clicked()
        {
            acts.push(Action::Forget);
        }
    });
}

/// The plan's fields (`sim::SamplingPlan`; the scan range stays at its default).
fn plan_fields(ui: &mut Ui, plan: &mut SamplingPlan) {
    egui::Grid::new("plan").num_columns(2).show(ui, |ui| {
        ui.label("perturbations");
        ui.add(egui::DragValue::new(&mut plan.perturbations).range(0..=2000).speed(1.0))
            .on_hover_text("random displacements of the scene, one teacher solve each");
        ui.end_row();
        ui.label("σ_r");
        ui.add(egui::DragValue::new(&mut plan.perturbation_sigma).range(0.0..=0.5).speed(0.002).fixed_decimals(3).suffix(" Bohr"))
            .on_hover_text("width of each coordinate's Gaussian displacement");
        ui.end_row();
        ui.label("hot burst");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut plan.hot_burst_steps).range(0..=5000).speed(1.0).suffix(" steps"));
            ui.add(egui::DragValue::new(&mut plan.hot_burst_t).range(0.0..=10000.0).speed(10.0).suffix(" K"));
        })
        .response
        .on_hover_text("a short thermostatted DFT-MD run from the scene (dt 0.25 fs, CSVR τ 5 fs); every step is a sample");
        ui.end_row();
        ui.label("scan points / pair");
        ui.add(egui::DragValue::new(&mut plan.scan_points_per_pair).range(0..=200).speed(0.2)).on_hover_text(format!(
            "each pair closer than the model's cutoff is scanned from {}× to {}× its current separation",
            plan.scan_range.0, plan.scan_range.1
        ));
        ui.end_row();
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut plan.seed).speed(0.2))
            .on_hover_text("advances after each pass, so a second pass draws new geometries");
        ui.end_row();
    });
}

/// The model's state: who answered this step, its uncertainty against the threshold (a bar with
/// the bound marked, and the numbers), trust (with the reason when not), samples, epoch, the last
/// fit and refit, fallbacks over the last 100 steps (bar), the teacher gap, the last audit and
/// the booked force-mismatch work.
fn readouts(ui: &mut Ui, d: &sim::DynSnapshot, m: &ModelStatus) {
    egui::Grid::new("model").num_columns(2).striped(true).max_col_width(230.0).show(ui, |ui| {
        // long values wrap in the dock instead of widening it
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        ui.label("this step");
        ui.colored_label(source_colour(d.source), RichText::new(source_label(d.source)).strong()).on_hover_text(SOURCE_TIP);
        ui.end_row();
        ui.label("uncertainty");
        let tip = "the model's force std per component (max over atoms) against the tightest per-atom bound min(max(2e-3, 10 % |F|), 1e-2) Ha/Bohr. A MODEL uncertainty: the surrogate inherits every DFT approximation (smearing, LSDA) on top of it.";
        match d.uncertainty {
            Some(u) => {
                let c = if u <= m.force_bound { GREEN } else { AMBER };
                ui.vertical(|ui| {
                    uncertainty_bar(ui, u, m.force_bound, c);
                    ui.colored_label(c, RichText::new(format!("{u:.1e} / {:.1e} Ha/Bohr", m.force_bound)).monospace());
                })
                .response
                .on_hover_text(tip);
            }
            None => {
                ui.colored_label(Color32::GRAY, "— (model not consulted)").on_hover_text(tip);
            }
        }
        ui.end_row();
        ui.label("trusted");
        let (c, text, ttip) = trust(Some(m));
        let r = ui.colored_label(c, text);
        if let Some(t) = ttip {
            r.on_hover_text(t);
        }
        ui.end_row();
        ui.label("samples");
        ui.monospace(format!("{}{}", m.samples, if m.pending_refit { " (refit pending)" } else { "" }));
        ui.end_row();
        ui.label("model epoch");
        ui.monospace(m.epoch.to_string()).on_hover_text("bumped whenever a refit moves the energy surface (the jump is booked into E conserved)");
        ui.end_row();
        ui.label("last fit");
        match m.fit {
            Some((n, q, e, f)) => ui.monospace(format!("{n} samples, Q′ {q}: rmse {e:.1e} Ha/atom, {f:.1e} Ha/Bohr")),
            None => ui.colored_label(Color32::GRAY, "—"),
        };
        ui.end_row();
        if let Some(r) = &m.last_refit {
            ui.label("last refit");
            match &r.result {
                Ok(n) => ui.monospace(format!("{n} samples, {:.0} ms, at step {} ({} this run)", r.ms, r.at_step, m.refits)),
                Err(e) => ui.colored_label(AMBER, format!("failed at step {}: {e}", r.at_step)),
            };
            ui.end_row();
        }
        ui.label("fallbacks");
        ui.vertical(|ui| {
            let f = d.fallback_fraction_100 as f32;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(200.0), 6.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 3.0, theme::PANEL_3);
            let mut fill = rect;
            fill.set_width(rect.width() * f.clamp(0.0, 1.0));
            ui.painter().rect_filled(fill, 3.0, AMBER);
            ui.monospace(format!("{:.0} % of the last 100 steps", 100.0 * d.fallback_fraction_100));
        });
        ui.end_row();
        ui.label("teacher gap");
        match (d.teacher_gap, m.last_gap) {
            (Some(g), _) => ui.monospace(format!("{g:+.2e} Ha (this step)")),
            (None, Some((g, step))) => ui.monospace(format!("{g:+.2e} Ha at step {step}")),
            (None, None) => ui.colored_label(Color32::GRAY, "— (no fallback on the model's surface yet)"),
        }
        .on_hover_text("E_DFT − E_model on the most recent fallback or audit booked on the model's surface (the energy stays on the model's surface; the gap is the model's error there)");
        ui.end_row();
        ui.label("last audit");
        match (&m.last_audit, m.audit_in_flight) {
            (None, true) => ui.colored_label(Color32::GRAY, "in flight on the background teacher"),
            (Some(a), in_flight) => {
                let (c, verdict) = match a.verdict {
                    sim::surrogate_ks::AuditVerdict::Passed => (GREEN, "passed"),
                    sim::surrogate_ks::AuditVerdict::Failed => (AMBER, "failed"),
                    sim::surrogate_ks::AuditVerdict::BoxAffected => (AMBER, "not judged (box-affected)"),
                };
                let lag = if a.asynchronous { format!("; audit lag: {} steps", a.unaudited_steps) } else { String::new() };
                let next = if in_flight { "; next in flight" } else { "" };
                ui.colored_label(c, format!("{verdict}: {:+.2e} Ha ({:.0} ms DFT{lag}{next})", a.gap, a.teacher_ms))
            }
            (None, false) => ui.colored_label(Color32::GRAY, "— (none yet)"),
        }
        .on_hover_text(if m.audits_background {
            "E_DFT − E_model at the last audited geometry. The audit runs on a second GPU teacher in the background while the model keeps answering (ruling S14); a failed one untrusts the model when it arrives. Audit lag = answered steps between the audit's dispatch and its verdict: those steps were answered unaudited. At full speed the model outruns the background teacher, so audits run back to back — one per teacher solve, whatever the 2000-step cadence says — and the lag is tens of thousands of steps. A box-affected teacher answer (density on the outermost grid plane > 2e-5 × (h / 0.25 Bohr)² e/Bohr³) carries the box's confinement and is not judged."
        } else {
            "E_DFT − E_model at the last audited geometry. With the CPU teacher the audit runs in the step (a second CPU solver would compete for the same cores). A box-affected teacher answer (density on the outermost grid plane > 2e-5 × (h / 0.25 Bohr)² e/Bohr³) carries the box's confinement and is not judged."
        });
        ui.end_row();
        ui.label("mismatch work");
        ui.monospace(format!("{:+.2e} Ha", d.work_mismatch))
            .on_hover_text("work of (F_DFT − F_model) on fallback steps, booked into E conserved so the drift measures the integrator, not the model");
        ui.end_row();
    });
}

/// The uncertainty against its bound: the bound sits at the middle of the bar, the value's dot
/// on a scale to twice the bound.
fn uncertainty_bar(ui: &mut Ui, u: f64, bound: f64, c: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(200.0), 10.0), egui::Sense::hover());
    let p = ui.painter();
    let mid = rect.left() + rect.width() / 2.0;
    let left = egui::Rect::from_min_max(rect.left_top(), egui::pos2(mid, rect.bottom()));
    let right = egui::Rect::from_min_max(egui::pos2(mid, rect.top()), rect.right_bottom());
    p.rect_filled(left, 4.0, GREEN.gamma_multiply(0.25));
    p.rect_filled(right, 4.0, AMBER.gamma_multiply(0.22));
    p.line_segment([egui::pos2(mid, rect.top() - 2.0), egui::pos2(mid, rect.bottom() + 2.0)], egui::Stroke::new(2.0, theme::INK));
    let x = if bound > 0.0 { (u / (2.0 * bound)).clamp(0.0, 1.0) as f32 } else { 1.0 };
    p.circle_filled(egui::pos2(rect.left() + x * rect.width(), rect.center().y), 5.0, c);
}
