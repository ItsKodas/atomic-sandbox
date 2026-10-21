//! The Model tab (the estimated solver, spec §6; rows 10–45 of the mapping): the honesty line,
//! the model's state per snapshot (who answered, uncertainty against its bound, trust, samples,
//! epoch, fits, fallbacks, teacher gap, last audit, mismatch work), the warning slots, the
//! sampling pass (progress, last pass, the long first-pass offer), the audit cadence, the
//! sampling plan, Start / Sample more / Cancel, and the model file.
use super::super::actions::{expire_arming, forget_armed, sample_button, SampleButton};
use super::super::sampling_presets as presets;
use super::super::{
    duration_label, search, source_colour, source_label, theme, trust, Action, PanelInput, UiState, AMBER, GREEN, RED, SOURCE_TIP,
};
use crate::scene::model_path;
use crate::sim::{source_line_with, Backend, ModelStatus, FIT_VS_HELD_HELP, IMPORT_REFERENCES_NOTE};
use egui::{Color32, RichText, Ui};
use sim::SamplingPlan;

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    expire_arming(st);
    ui.horizontal_wrapped(|ui| {
        theme::note(ui, "estimated = a model fitted to this scene's DFT; valid only where it has been taught — it falls back to DFT when unsure");
        theme::info(ui, "The estimated solver is a model of this DFT, not of nature: it inherits every approximation of its teacher (smearing, LSDA, the box) and adds its own fit error. Help › Physics, honestly has the long form.");
    });
    resume_row(ui, st, input, acts);
    if st.backend != Backend::Estimated {
        theme::note(ui, "The estimated solver is not running: pick Estimated in the toolbar's Solver group. Its model, sampling pass and audits show here.");
        return;
    }
    let snap = input.snapshot.filter(|s| s.backend == Backend::Estimated);
    let d = snap.map(|s| &s.dynamics);
    let m: Option<&ModelStatus> = input.model();
    let sampling = input.sampling();
    super::group(ui, st, "d-model-source", |ui, st| source_row(ui, st, m, sampling, acts));
    super::group(ui, st, "d-model", |ui, st| {
        if let (Some(d), Some(m)) = (d, m) {
            readouts(ui, d, m, input.scene, st);
        }
    });
    if m.is_some_and(|m| crate::sim::box_blocks_trust(m).is_some()) {
        warnbox(ui, &super::super::box_advice(input.scene, st));
    }
    if let Some(m) = m {
        no_model_row(ui, m, sampling, acts);
        teach_row(ui, m, sampling, acts);
    }
    if let Some(m) = m {
        // the model's info line (M6 Task 5's parked item): what it was taught at, and the mass
        // set recorded with it — recorded, never compared (ruling P2)
        if let Some(p) = &m.provenance {
            theme::note(ui, format!("taught at {p}"))
                .on_hover_text("the settings a model is valid at (grid, h, σ, charge, multiplicity, teacher and its tolerance — a scene at others falls back to DFT), and the isotopes its samples were taken with: recorded, not compared, because the Born-Oppenheimer surface does not depend on the nuclear masses");
        }
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
        // an on/off switch beside the cadence (user request 2026-10-18), with why they matter
        let mut on = st.audit_every > 0;
        if super::super::probed("audits-on", ui.checkbox(&mut on, "audits on"))
            .on_hover_text(AUDIT_WHY)
            .changed()
        {
            if on {
                acts.push(Action::SetAudit(st.audit_every_on.max(1)));
            } else {
                st.audit_every_on = st.audit_every.max(1);
                acts.push(Action::SetAudit(0));
            }
        }
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
            super::super::probed("audit-auto", ui.checkbox(&mut st.audit_auto, "auto"))
                .on_hover_text(format!(
                    "set the cadence from the play speed so an audit comes about every {:.0} s of wall time at any speed (e.g. ≈ 15 fs/s at dt 0.25 fs → every ≈ 600 steps, raised to the floor; ≈ 4000 fs/s → every ≈ 160 000). Faster play checks a smaller share of steps but keeps the same steady load on the GPU; the cadence stays between {} and {} steps",
                    super::super::AUTO_AUDIT_WALL_S, super::super::AUTO_AUDIT_MIN, super::super::AUTO_AUDIT_MAX
                ));
            // what the worker runs (after a panic recovery it re-applies this value)
            if let Some(m) = m.filter(|m| m.audit_every != st.audit_every) {
                ui.colored_label(AMBER, format!("(solver: {})", m.audit_every));
            }
        });
        if st.audit_every == 0 {
            ui.colored_label(AMBER, "audits off: nothing checks the model against DFT while it answers (unsafe)");
            ui.label(RichText::new(AUDIT_OFF_NOTE).size(11.0).color(theme::INK_DIM));
        }
    });
    let fitted = m.is_some_and(|m| m.fitted);
    let r = egui::CollapsingHeader::new("Sampling plan")
        .id_salt("sampling-plan")
        .default_open(!fitted)
        .open(search::opens(st, "d-plan"))
        .show(ui, |ui| {
            let tu = st.temp_units;
            plan_fields(ui, &mut st.plan, tu)
        });
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
    // Q1 of Appendix B (M5b): draft mode, also in Physics ▸ Estimated solver
    super::group(ui, st, "d-draft", |ui, st| {
        theme::section(ui, "Draft mode");
        super::super::menus::draft_control(ui, st, acts);
    });
}

/// Where the model learns (M8a Task 4b): the source line ([`source_line`]), the switch between the
/// shared collection and a model file (offered when a collections folder is set), and "Import a
/// model file…" with the last import's outcome.
fn source_row(ui: &mut Ui, st: &mut UiState, m: Option<&ModelStatus>, sampling: bool, acts: &mut Vec<Action>) {
    let Some(m) = m else { return };
    // the switch shows the user's choice (a charged scene learns per-scene under it); Import needs
    // the collection attached
    let shared = m.use_collection;
    let attached = matches!(m.source, sim::ModelSource::Collection { .. });
    let read_only = matches!(m.source, sim::ModelSource::Collection { read_only: true, .. });
    let line = RichText::new(source_line_with(m, st.model_file.as_deref())).size(12.5).color(if read_only { AMBER } else { theme::INK });
    super::super::probed("model-source", ui.label(line)).on_hover_text(SOURCE_LINE_TIP);
    if !m.collections {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(!sampling, egui::RadioButton::new(shared, "shared collection")).on_hover_text("learn into the one model of this solver surface (h, σ, ionic potential, teacher): every scene and box at these settings adds to it, and it saves itself as it learns (models/collections)").clicked() && !shared {
            acts.push(Action::UseCollection);
        }
        if ui.add_enabled(!sampling, egui::RadioButton::new(!shared, "a model file…")).on_hover_text("switch to a model file as a specialised model for this scene (per-scene; Save model writes it)").clicked() {
            let last = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
            if let Some(p) = model_dialog(&last).pick_file() {
                st.model_file = Some(p);
                acts.push(Action::LoadModel);
            }
        }
    });
    let can_import = attached && !read_only && !sampling;
    if super::super::probed("import-model", ui.add_enabled(can_import, egui::Button::new("Import a model file…")))
        .on_disabled_hover_text("needs the shared collection attached and writable, and no pass running")
        .on_hover_text(format!("add a .model file's samples to the shared collection — each one checked (the same solver surface, box status known, not a duplicate); what is refused is listed with the reason. {IMPORT_REFERENCES_NOTE}"))
        .clicked()
    {
        let last = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
        if let Some(p) = model_dialog(&last).pick_file() {
            acts.push(Action::ImportModel(p));
        }
    }
    match &m.last_import {
        Some(Ok(r)) => {
            ui.label(RichText::new(import_line(r)).size(11.5));
            for (i, why) in r.refused.iter().take(5) {
                ui.weak(format!("  {}: {why}", i.map_or("the file".to_string(), |i| format!("sample #{i}"))));
            }
            if r.refused.len() > 5 {
                ui.weak(format!("  … and {} more refused", r.refused.len() - 5));
            }
            for n in r.notes.iter().take(3) {
                ui.weak(n);
            }
            ui.label(RichText::new(IMPORT_REFERENCES_NOTE).size(11.0).color(theme::INK_DIM));
        }
        Some(Err(e)) => warnbox(ui, &format!("import refused: {e}")),
        None => {}
    }
}

/// "imported N, refused M" of an import (the Model tab).
pub fn import_line(r: &sim::ImportSummary) -> String {
    format!("last import: imported {}, refused {}", r.imported, r.refused.len())
}

/// The source line's hover text.
const SOURCE_LINE_TIP: &str = "where the estimated solver's model comes from: the shared collection of this solver surface (one folder in models/collections per h, σ, ionic potential, teacher and tolerance — every scene at those settings learns into it), a model file (a specialised per-scene model), or this scene only (a charged scene or a forced multiplicity keeps a per-scene model: collections are neutral until M9)";

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

/// Samples held but no model formed (`ModelStatus::no_model`; the 2026-10-20 PH₃ session): the
/// reason in plain words and, when it can help, **Retry** (re-solves a missing isolated-atom
/// reference, or refits — `Action::RetryFit`). Also in the Activity tab.
pub fn no_model_row(ui: &mut Ui, m: &ModelStatus, sampling: bool, acts: &mut Vec<Action>) {
    let Some(why) = &m.no_model else { return };
    warnbox(ui, why);
    if m.retry_fit || sampling {
        ui.horizontal_wrapped(|ui| {
            let b = egui::Button::new(RichText::new("Retry").strong().color(Color32::from_rgb(0x1c, 0x14, 0x05))).fill(AMBER);
            if super::super::probed("retry-fit", ui.add_enabled(!sampling && m.retry_fit, b))
                .on_disabled_hover_text("a pass is running")
                .on_hover_text("solve the missing isolated-atom reference — every allowed multiplicity, each from a cold start (a solve lost to a busy GPU is tried once more) — then fit the samples held (a short pass: pauses the dynamics). With every reference known it fits at once. Not offered when the atom's SCF itself failed: a cold solve is deterministic and would fail the same way.")
                .clicked()
            {
                acts.push(Action::RetryFit);
            }
            ui.weak("the samples are kept either way: Save model… writes them");
        });
    }
}

/// The running pass's progress bar and counts (also on the sampling card in the 3D view).
pub fn progress(ui: &mut Ui, p: &sim::SamplingProgress, m: Option<&ModelStatus>) {
    let frac = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
    let eta = m.and_then(|m| m.pass_eta_s).map_or(String::new(), |t| format!(" — about {} left", duration_label(t)));
    ui.add(egui::ProgressBar::new(frac).show_percentage().text(format!("{} — {}/{}{eta}", crate::ui::activity::words::pass_stage(p), p.done, p.total)));
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
            "a first sampling pass here: ≈ {} at this scene's first solve ({} solves × {:.1} s; the panel's plan, {} solves, ≈ {}); a pass's own solves differ (references, warm starts) — the time left is shown once it runs. That is already the fewest solves that can earn trust on this scene. A smaller grid is faster.",
            duration_label(o.eta_s),
            o.solves,
            o.per_solve_s,
            o.full_solves,
            duration_label(o.full_eta_s)
        ),
    );
    if ui.button(RichText::new(format!("Start anyway (≈ {} at the first solve)", duration_label(o.eta_s))).color(AMBER)).clicked() {
        acts.push(Action::StartOffered(o.plan.clone()));
    }
}

/// The teach-around button's label, "Teach around N blockers (≈M solves, ≈T)", T at the model's
/// solve time; `None` when nothing is teachable (`ModelStatus::teach_offer`).
pub fn teach_label(m: &ModelStatus) -> Option<String> {
    let o = m.teach_offer.as_ref().filter(|o| !o.blockers.is_empty())?;
    let eta = m.solve_s.map(|s| format!(", ≈{}", super::super::activity::headline::clock(s * o.solves as f64))).unwrap_or_default();
    Some(format!("Teach around {} blocker{} (≈{} solves{eta})", o.blockers.len(), if o.blockers.len() == 1 { "" } else { "s" }, o.solves))
}

/// "Teach around N blockers (≈M solves, ≈T)" when an environment class is blocked by training
/// samples the rest of the data cannot predict (`ModelStatus::teach_offer`), with the blockers the
/// pass would not teach and why. The Model tab and the Activity tab's trust checklist both show it.
pub fn teach_row(ui: &mut Ui, m: &ModelStatus, sampling: bool, acts: &mut Vec<Action>) {
    let Some(o) = &m.teach_offer else { return };
    // the warning and its fix in one amber box whose outline pulses (user request 2026-10-19:
    // the button was easy to miss); still while a pass runs
    let pulse = if sampling {
        0.0
    } else {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(40));
        (0.5 + 0.5 * (ui.input(|i| i.time) * 3.5).sin()) as f32
    };
    egui::Frame::new()
        .fill(AMBER.gamma_multiply(0.08))
        .stroke(egui::Stroke::new(2.0, AMBER.gamma_multiply(0.25 + 0.75 * pulse))) // fixed width: only the brightness pulses (a width change moved the layout)
        .corner_radius(8)
        .inner_margin(10)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if let Some(p) = &o.prompt {
                ui.colored_label(AMBER, p);
            }
            if let Some(label) = teach_label(m) {
                let names: Vec<String> = o.blockers.iter().map(|(i, c, loo, ins)| format!("#{i} ({c}: leave-one-out {loo:.1}×, in-sample {ins:.2}×)")).collect();
                ui.add_space(4.0);
                let button = egui::Button::new(RichText::new(format!("Fix it: {label}")).strong().size(14.0).color(egui::Color32::from_rgb(0x1c, 0x14, 0x05)))
                    .fill(AMBER)
                    .corner_radius(6)
                    .min_size(egui::Vec2::new(ui.available_width().min(360.0), 32.0));
                if ui
                    .add_enabled(!sampling, button)
                    .on_disabled_hover_text("a pass is running")
                    .on_hover_text(format!(
                        "a targeted pass around the training samples that keep a class untrusted: {}. About 10 DFT solves each — random nudges each clearly apart from the sample and from each other, and steps toward its nearest learned neighbour — of which 2 are held out to check the result. Accepted only if the unchanged class rule then passes with the sample still in the fit and the checks pass; at most two rounds. Pauses the dynamics.",
                        names.join(", ")
                    ))
                    .clicked()
                {
                    acts.push(Action::TeachAroundBlockers);
                }
            }
        });
    for (i, c, ins, why) in &o.refused {
        ui.weak(format!("#{i} ({c}, in-sample {ins:.2}×): not taught — {why}"));
    }
    if o.refused_total > o.refused.len() {
        ui.weak(format!("… and {} more not taught", o.refused_total - o.refused.len()));
    }
    if !o.blockers.is_empty() {
        for n in &o.notes {
            ui.weak(n);
        }
    }
}

/// Start sampling / Sample more (armed "Start anyway" above 2 min), the plan's cost, Cancel.
fn sample_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let sampling = input.sampling();
    // an empty scene has no species to sample (`sim::sampling::EMPTY_SCENE`)
    let empty = input.scene.nuclei.is_empty();
    let SampleButton { label, long, armed, cost } = sample_button(st, input);
    ui.horizontal_wrapped(|ui| {
        let text = if armed { RichText::new(label).color(AMBER) } else { RichText::new(label).strong() };
        let b = if armed { egui::Button::new(text) } else { egui::Button::new(text.color(Color32::from_rgb(0x17, 0x11, 0x2a))).fill(theme::ACCENT) };
        if super::super::probed("sample", ui.add_enabled(!sampling && !empty, b))
            .on_disabled_hover_text(if empty { super::super::EMPTY_SAMPLE } else { "a pass is running" })
            .on_hover_text("one pass over the current scene with the plan above: isolated-atom references, random perturbations, pair scans, a hot MD burst; the samples are added to the model's and it is refitted. Pauses the dynamics. A pass longer than 2 min asks for a second click — estimated as solves × the last pass's measured rate, or × this scene's first solve before any pass (an estimate, not a bound: a pass's own solves differ; its time left is shown once it runs).")
            .clicked()
        {
            acts.push(Action::Sample { long });
        }
        if empty {
            ui.weak(super::super::EMPTY_SAMPLE);
        } else if let Some(c) = cost {
            ui.weak(c);
        }
        if ui
            .add_enabled(sampling, egui::Button::new("Cancel"))
            .on_hover_text("end the pass now: what it sampled is kept; it is refitted only once there are enough samples to fit (Q′, the model's number of directions), else the Log says so and Sample more adds to them")
            .clicked()
        {
            acts.push(Action::CancelSampling);
        }
    });
}

/// Save model / Load model / Forget model (two clicks within 4 s): File ▸ Estimated model and
/// the Model tab.
/// Why audits matter, for the "audits on" switch's hover (user request 2026-10-18).
pub const AUDIT_WHY: &str = "Audits are the model's spot checks: every n-th answered step a real DFT solve is run at the same geometry and compared with the model's answer. They are how the app notices that the model has wandered somewhere it was never taught and is quietly answering wrong — a failed audit makes the model untrusted and hands the steps back to DFT, and the audit's answer becomes a training sample. Turning them off can make playback smoother (on the GPU the background audit competes with the display for the card, which can show as a stutter every so often), but nothing then checks the model while it answers: a wrong answer looks exactly like a right one. Keep them on unless you are only watching a geometry the model has seen many times.";

/// Shown under the cadence while audits are off.
pub const AUDIT_OFF_NOTE: &str = "Smoother, but blind: the model's answers are no longer checked, so errors outside what it learned go unnoticed. Turn audits back on before trusting what you see (the last cadence is restored).";

/// The native Save/Open dialog for model files, starting in `last`'s folder with its name
/// (user request 2026-10-18: pick the save location and the file to load); a bare name (nothing
/// chosen yet) starts in the dist folder's `models` (`scene::models_dir`) when there is one.
pub(in crate::ui) fn model_dialog(last: &std::path::Path) -> rfd::FileDialog {
    let mut d = rfd::FileDialog::new().add_filter("Atomic Sandbox model", &["model"]).add_filter("All files", &["*"]);
    let dir = last.parent().filter(|p| !p.as_os_str().is_empty()).map(std::path::Path::to_path_buf).or_else(crate::scene::models_dir).or_else(|| std::env::current_dir().ok());
    if let Some(dir) = dir {
        d = d.set_directory(dir);
    }
    if let Some(name) = last.file_name() {
        d = d.set_file_name(name.to_string_lossy());
    }
    d
}

pub fn model_file_buttons(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    expire_arming(st);
    let estimated = st.backend == Backend::Estimated;
    let m = input.model();
    let sampling = input.sampling();
    let fitted = m.is_some_and(|m| m.fitted);
    // samples without a fit are saved too (a samples-only file): a fit that did not form must not
    // lose the DFT taught so far
    let savable = fitted || m.is_some_and(|m| m.samples > 0);
    let path = st.model_file.clone().unwrap_or_else(|| model_path(&st.scene_path));
    ui.horizontal_wrapped(|ui| {
        let tip = if fitted {
            format!("choose where to write the model, its fit and its samples (last: {})", path.display())
        } else {
            format!("no model yet: choose where to write the samples taught so far — loading the file fits them once a fit can form (last: {})", path.display())
        };
        if ui
            .add_enabled(estimated && savable, egui::Button::new("Save model…"))
            .on_disabled_hover_text("nothing to save yet: no samples")
            .on_hover_text(tip)
            .clicked()
        {
            // the File window (`ui::files`): the models folder, with the native dialog one click away there
            super::super::files::open_save_model(st);
        }
        if ui
            .add_enabled(estimated && !sampling, egui::Button::new("Load model…"))
            .on_hover_text(
                "choose a .model file to read — it only runs on the grid, σ, charge, multiplicity, ionic potential, teacher and teacher tolerance it was trained at; if the scene differs, a card lists the differences and offers to apply the model's settings",
            )
            .clicked()
        {
            super::super::files::open(st);
        }
        let armed = forget_armed(st);
        let samples = m.map_or(0, |m| m.samples);
        // in collection mode only this session's own samples go: the collection's are permanent
        let in_collection = m.and_then(|m| match m.source {
            sim::ModelSource::Collection { samples: k, .. } => Some(k),
            _ => None,
        });
        let forget = match (armed, in_collection) {
            (true, Some(k)) => egui::Button::new(RichText::new(format!("Click again to forget {} session-only samples", samples.saturating_sub(k))).color(RED)),
            (true, None) => egui::Button::new(RichText::new(format!("Click again to forget {samples} samples")).color(RED)),
            (false, _) => egui::Button::new(RichText::new("Forget model…").color(theme::DANGER_INK)),
        };
        let tip = if in_collection.is_some() {
            "drop this session's own samples — the ones the shared collection did not take (two clicks within 4 s); the shared collection stays attached and keeps its samples: they are permanent, nothing forgets them"
        } else {
            "drop the model, its samples and references (two clicks within 4 s); the teacher answers every step until the next pass"
        };
        if super::super::probed("forget", ui.add_enabled(estimated && (m.is_some_and(|m| m.samples > 0 || m.fitted) || sampling), forget))
            .on_hover_text(tip)
            .clicked()
        {
            acts.push(Action::Forget);
        }
    });
}

/// The preset row (`ui/sampling_presets.rs`): picking one sets the plan's fields and keeps its
/// seed; the selection is read off the plan, so an edited field shows "Custom". The selected
/// preset's description sits under the row.
fn preset_row(ui: &mut Ui, plan: &mut SamplingPlan) {
    let current = presets::matching(plan);
    ui.horizontal(|ui| {
        ui.label("preset").on_hover_text(presets::RULE_OF_THUMB);
        egui::ComboBox::from_id_salt("plan-preset")
            .selected_text(current.map_or(presets::CUSTOM, |i| presets::PRESETS[i].name))
            .show_ui(ui, |ui| {
                for (i, p) in presets::PRESETS.iter().enumerate() {
                    if ui.selectable_label(current == Some(i), p.name).on_hover_text(p.description).clicked() {
                        p.apply(plan);
                    }
                }
            })
            .response
            .on_hover_text(presets::RULE_OF_THUMB);
    });
    if let Some(i) = presets::matching(plan) {
        theme::note(ui, presets::PRESETS[i].description);
    }
}

/// The plan's fields (`sim::SamplingPlan`; the scan range stays at its default), under the
/// preset row.
fn plan_fields(ui: &mut Ui, plan: &mut SamplingPlan, tu: super::super::TempUnits) {
    preset_row(ui, plan);
    super::wide_grid(ui, "plan", 2).show(ui, |ui| {
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
            // typed in the chosen scale, kept in kelvin (clamped at absolute zero)
            let mut shown = tu.from_k(plan.hot_burst_t);
            if ui.add(egui::DragValue::new(&mut shown).range(tu.range(0.0, 10000.0)).speed(10.0).max_decimals(2).suffix(tu.suffix())).changed() {
                plan.hot_burst_t = tu.to_k(shown);
            }
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

/// The hover text of the "learned classes" rows.
pub const CLASS_LOO_HELP: &str = "after every refit each training sample is left out (exactly, in closed form) and the fit without it must predict it within the force bound and the energy tier — separately for \"pair\" environments (every atom with at most one neighbour within the model's cutoff) and \"many-body\" ones. The numbers are the worst left-out errors as multiples of those bounds (1× = at the bound). A class that fails, or holds fewer samples than the model has directions, is not answered by the model: its steps fall back to DFT. Until DFT has checked the model at three or more geometries of a class, one badly predicted sample is enough to keep the whole class on DFT; from then on the class is judged by the error those DFT checks measured (on average, with a margin to start answering), with the left-out average as a floor.";

/// One class's row: "pair: trusted · 273 samples · left-out force ≤ 0.10× · energy ≤ 0.15×".
pub fn class_line(c: &sim::ClassLoo) -> String {
    format!(
        "{}: {} · {} samples · left-out force ≤ {:.2}× · energy ≤ {:.2}×",
        c.class,
        if c.trusted { "trusted" } else { "not learned yet (DFT)" },
        c.samples,
        c.force_max,
        c.energy_max
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_class_row_names_the_class_its_trust_and_its_left_out_errors() {
        let c = sim::ClassLoo { class: "many-body", samples: 41, force_rms: 0.9, force_max: 2.69, energy_rms: 0.3, energy_max: 0.52, trusted: false, refusal: Some((sim::ClassKey::Force, 2.69, 1.0)), calibration: 1.0, measured: None, measured_energy: None, worst_atom: None, evidence: 0, trusted_since: None, inflation: 1.0 };
        assert_eq!(super::class_line(&c), "many-body: not learned yet (DFT) · 41 samples · left-out force ≤ 2.69× · energy ≤ 0.52×");
    }
}

/// The model's state: who answered this step, its uncertainty against the threshold (a bar with
/// the bound marked, and the numbers), trust (with the reason when not), samples, epoch, the last
/// fit and refit, fallbacks over the last 100 steps (bar), the teacher gap, the last audit and
/// the booked force-mismatch work.
fn readouts(ui: &mut Ui, d: &sim::DynSnapshot, m: &ModelStatus, scene: &crate::scene::Scene, st: &UiState) {
    super::wide_grid(ui, "model", 2).striped(true).show(ui, |ui| {
        // long values wrap in the dock instead of widening it
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        ui.label("this step");
        ui.colored_label(source_colour(d.source), RichText::new(source_label(d.source)).strong()).on_hover_text(SOURCE_TIP);
        ui.end_row();
        ui.label("uncertainty");
        let tip = "the model's epistemic force std per component (max over atoms; the label noise σ_f is left out — a fallback on it would teach nothing, and a fit whose σ_f exceeds 2e-3 Ha/Bohr is untrusted instead) against the tightest per-atom bound min(max(2e-3, 10 % |F|), 1e-2) Ha/Bohr. A MODEL uncertainty: the surrogate inherits every DFT approximation (smearing, LSDA) on top of it.";
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
        let t = trust(Some(m), scene, st);
        let r = ui.colored_label(t.colour, t.text);
        // the plain line; then what it means, then the provider's own text
        let hover = [t.tip.map(str::to_string), t.detail.map(|d| format!("the fit's own words: {d}"))].into_iter().flatten().collect::<Vec<_>>().join("

");
        if !hover.is_empty() {
            r.on_hover_text(hover);
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
        if let Some(r) = &m.readiness {
            ui.label("strict here");
            let text = if r.answers { "answers".to_string() } else { format!("refuses — {}", r.text) };
            ui.colored_label(if r.answers { GREEN } else { AMBER }, text).on_hover_text("the fallback policy's verdict at this geometry with draft mode off: whether the model would answer a step here, else the rule that sends it to DFT");
            ui.end_row();
            if let Some(u) = &r.unlearned {
                ui.label("this scene");
                ui.colored_label(AMBER, u);
                ui.end_row();
            }
        }
        if let Some(text) = &m.fit_vs_held {
            ui.label("fit vs held");
            ui.monospace(text).on_hover_text(FIT_VS_HELD_HELP);
            ui.end_row();
        }
        if !m.class_loo.is_empty() {
            // final review app M4: the class-wise trust behind "not learned well enough yet"
            ui.label("learned classes");
            ui.vertical(|ui| {
                for c in &m.class_loo {
                    ui.colored_label(if c.trusted { GREEN } else { AMBER }, RichText::new(class_line(c)).monospace());
                }
            })
            .response
            .on_hover_text(CLASS_LOO_HELP);
            ui.end_row();
        }
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
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
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
        // the step it was judged at (the driver's stamp on its event; activity Task 5): for a
        // background audit that is after its dispatch, by the lag in answered steps
        let at = m.last_audit_step.map_or(String::new(), |n| format!(" at step {n}"));
        match (&m.last_audit, m.audit_in_flight) {
            (None, true) => ui.colored_label(Color32::GRAY, "in flight on the background teacher"),
            (Some(a), in_flight) => {
                let (c, verdict) = match a.verdict {
                    sim::AuditVerdict::Passed => (GREEN, "passed"),
                    sim::AuditVerdict::Failed => (AMBER, "failed"),
                    sim::AuditVerdict::BoxAffected => (AMBER, "not judged (box-affected)"),
                };
                let lag = if a.asynchronous { format!("; audit lag: {} answered steps", a.unaudited_steps) } else { String::new() };
                let next = if in_flight { "; next in flight" } else { "" };
                ui.colored_label(c, format!("{verdict}: model off by {:+.2e} Ha{at} ({:.0} ms DFT{lag}{next})", a.gap, a.teacher_ms))
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
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 10.0), egui::Sense::hover());
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

/// The interrupted training's offer (`UiState::resume`; user report 2026-10-20): resume the pass
/// where it stopped, on its own scene and in the estimated solver, or discard it. Shown on every
/// solver; hidden while a pass runs (a pass must end before another resumes).
fn resume_row(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if input.sampling() {
        // a running pass owns the folder now (a resumed one, or a new one replacing it)
        st.resume = None;
        return;
    }
    let Some(o) = st.resume.as_ref() else { return };
    let z: Vec<u8> = o.scene.nuclei.iter().map(|n| n.z).collect();
    let what = crate::names::formula(&z, (o.scene.charge != 0).then_some(o.scene.charge));
    egui::Frame::new().fill(AMBER.gamma_multiply(0.08)).stroke(egui::Stroke::new(1.5, AMBER.gamma_multiply(0.6))).inner_margin(6.0).corner_radius(4.0).show(ui, |ui| {
        ui.colored_label(
            AMBER,
            format!(
                "An interrupted training: {what} at {}³ × {} Bohr, sample {} of {} ({}). Its samples are on disk.",
                o.scene.grid_n,
                o.scene.grid_h,
                (o.done + 1).min(o.total),
                o.total,
                o.stage
            ),
        );
        ui.horizontal(|ui| {
            let resume = ui
                .button(RichText::new(format!("Resume the interrupted training (sample {} of {})", (o.done + 1).min(o.total), o.total)).color(AMBER))
                .on_hover_text("switches to the estimated solver on the pass's own scene (its grid, σ, charge and ionic potential), loads the samples it had taught and goes on with its remaining jobs — the current scene and model are replaced");
            if resume.clicked() {
                acts.push(Action::ResumeTraining);
            }
            if ui.button("Discard").on_hover_text("removes the interrupted pass's files; a new sampling pass also replaces them").clicked() {
                acts.push(Action::DiscardTraining);
            }
        });
    });
}
