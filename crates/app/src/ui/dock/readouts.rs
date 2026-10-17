//! The Readouts tab: T, drift and the measured rate as tiles; the energies (Ha, eV or both)
//! with the work rows; time and cost; a small E total / T sparkline; health (box, dt and
//! evaluation warnings, the newest dynamics notes); and the collapsible energy breakdown and
//! solver diagnostics (rows 51–72 and 111–127 of the mapping).
use super::super::pseudo;
use super::super::{
    box_warning, drift_colour, energy_cells, search, theme, GraphSeries, PanelInput, UiState, AMBER, BOX_MARGIN_WARN, GREEN,
    SPARK_WINDOW,
};
use super::graph::sparkline;
use crate::sim::Backend;
use egui::{Color32, RichText, Ui};
use sim::dynamics::{DRIFT_MIN_POINTS, DT_WARN_FS, F_DRAG_HI, F_DRAG_LO, F_DRAG_MAX};
use sim::Source;

fn tile(ui: &mut Ui, key: &str, value: RichText, unit: &str, tip: &str) {
    egui::Frame::new()
        .fill(theme::PANEL_2)
        .stroke(egui::Stroke::new(1.0, theme::LINE_SOFT))
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.set_width((ui.available_width()).max(70.0));
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.label(RichText::new(key.to_uppercase()).size(10.0).color(theme::INK_FAINT));
                ui.label(value.monospace().strong().size(14.0));
                ui.label(RichText::new(unit).size(10.5).color(theme::INK_DIM));
            });
        })
        .response
        .on_hover_text(tip);
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    let Some(s) = input.snapshot else {
        ui.weak("no solution yet");
        return;
    };
    let d = &s.dynamics;
    let estimated = s.backend == Backend::Estimated;
    let tu = st.temp_units;
    let t_shown = super::super::averaged_temperature_text(d, &st.temp_avg, 1, tu);
    super::group(ui, st, "d-tiles", |ui, _| {
        ui.columns(3, |c| {
            tile(
                &mut c[0],
                "T",
                RichText::new(t_shown.clone()),
                "1 s average",
                "kinetic temperature by equipartition over 3N − 3 degrees of freedom (COM motion removed; rotation counted; a single nucleus: 3, lab frame), averaged over the last second of steps — a few nuclei's instantaneous T swings with every vibration",
            );
            let (v, unit) = match d.drift_per_ps {
                Some(x) => {
                    let col = drift_colour(x).unwrap_or(theme::INK);
                    let word = match drift_colour(x) {
                        Some(c) if c == AMBER => "Ha/ps · amber",
                        Some(_) => "Ha/ps · red",
                        None => "Ha/ps · ok",
                    };
                    (RichText::new(format!("{x:+.2e}")).color(col), word)
                }
                None => (RichText::new("—").color(Color32::GRAY), "Ha/ps"),
            };
            tile(
                &mut c[1],
                "drift",
                v,
                unit,
                &format!(
                    "least-squares slope of E conserved over the last ≤ 2000 steps (shown after {DRIFT_MIN_POINTS}); amber above 1e-4, red above 1e-3 Ha/ps"
                ),
            );
            let rate = input.measured_fs_per_s.map_or("—".to_string(), |r| format!("{r:.1}"));
            tile(
                &mut c[2],
                "rate",
                RichText::new(rate),
                "fs/s, last 2 s",
                "simulated femtoseconds per wall second over the last 2 s of play (what you see)",
            );
        });
    });
    super::group(ui, st, "d-energies", |ui, st| {
        ui.horizontal(|ui| {
            theme::section(ui, "Energies");
            theme::info(ui, "E total = KE + E_KS: conserved in NVE without tools. E conserved = E total + drag spring + field potential + container walls − the work of thermostat, pins, drag damping, COM correction and moving walls.");
        });
        let units = st.energy_units;
        super::wide_grid(ui, "energies", 3).striped(true).show(ui, |ui| {
            let mut row = |name: &str, x: f64, strong: bool, tip: &str| {
                let n = if strong { RichText::new(name).strong() } else { RichText::new(name) };
                ui.label(n).on_hover_text(tip);
                energy_cells(ui, x, units, strong);
                ui.end_row();
            };
            row("KE (nuclei)", d.kinetic, false, "Σ ½ m v² of the nuclei (lab frame)");
            let e_ks_label = pseudo::energy_label(&d.eval.pseudised);
            let e_ks_tip = if d.eval.pseudised.is_empty() {
                "Kohn-Sham energy at the current positions (electrons + nuclear repulsion)".to_string()
            } else {
                format!("Kohn-Sham energy at the current positions (valence electrons + the ions' repulsion). {}", pseudo::VALENCE_NOTE)
            };
            row(e_ks_label, d.energy_ks, false, &e_ks_tip);
            row("E total", d.total, true, "KE + E_KS: conserved in NVE without tools");
            row(
                "E conserved",
                d.energy_conserved,
                false,
                "E total + drag spring + field potential + container walls − the work of the thermostat, pins, drag damping, COM correction and moving walls: the invariant the drift is fitted on",
            );
            let model_work = if estimated { d.work_refit + d.work_mismatch } else { 0.0 };
            row(
                "work done by tools",
                d.total - d.energy_conserved - model_work,
                false,
                &format!(
                    "the energy the drag, field, thermostat, pins, COM correction and container walls have put into the nuclei and electrons (net of what the spring, the field and the walls still hold). \
                     The drag spring pulls with at most the pull ({F_DRAG_MAX} Ha/Bohr by default; the wheel while dragging sets it, {F_DRAG_LO}–{F_DRAG_HI}): drag slowly and the molecule follows with little work; a fast drag mostly heats it (and every tug shows up here). \
                     With the estimated solver the model's own bookings are on the rows below; the three add up to E total − E conserved."
                ),
            );
            if d.walls.is_some() || d.walls_work != 0.0 {
                row(
                    "   of which walls (piston)",
                    d.walls_work,
                    false,
                    "the work of switching and moving the container walls (ΔU of the wall at the atoms' positions per move): positive when shrinking the box squeezes them",
                );
            }
            if estimated {
                row(
                    "   model refit",
                    d.work_refit,
                    false,
                    "the jumps of the model's energy surface under the atoms (a refit, a failed audit's switch to DFT), booked so E conserved stays continuous: a model learning a new bonding situation can book a large one. Not work anybody did — the surface the energy is measured on moved.",
                );
                row(
                    "   model force mismatch",
                    d.work_mismatch,
                    false,
                    "work of (F_DFT − F_model) on fallback steps: the atoms move on DFT forces while the energy is the model's, so the difference is booked like an external force",
                );
            }
        });
        if estimated {
            theme::note(ui, "The three work rows add up to E total − E conserved.");
        }
        // M7 (Task 6 review A.6): a partly filled degenerate shell is shared equally, the spherical-atom ensemble
        for line in pseudo::occupation_lines(&d.eval.fractional) {
            theme::note(ui, line);
        }
    });
    super::group(ui, st, "d-time", |ui, _| {
        theme::section(ui, "Time and cost");
        super::wide_grid(ui, "time-cost", 2).striped(true).show(ui, |ui| {
            // long values wrap in the dock instead of widening it
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            ui.label("sim time");
            ui.monospace(format!("t = {:.2} fs · step {}", d.sim_time_fs(), d.step));
            ui.end_row();
            ui.label("fs / s").on_hover_text("simulated femtoseconds per wall second: over the last 2 s of play (what you see), and over the last tick alone (a DFT step and a surrogate tick differ by 10⁴)");
            let measured = input.measured_fs_per_s.map_or("—".to_string(), |r| format!("{r:.1}"));
            ui.monospace(format!("{measured} (last 2 s) · {:.2} (this tick)", d.wall_ratio));
            ui.end_row();
            ui.label("last tick");
            ui.monospace(format!(
                "{} step{} · {:.3} ms/step",
                d.steps_last_tick,
                if d.steps_last_tick == 1 { "" } else { "s" },
                d.ms_per_step
            ));
            ui.end_row();
            ui.label("per step");
            if d.source == Source::Surrogate {
                ui.monospace(format!("model, no SCF · {:.3} ms", d.ms_per_step_model)).on_hover_text(format!(
                    "moving mean over the model's own steps; DFT steps here (fallbacks; audits in the step with the CPU teacher) average {:.0} ms",
                    d.ms_per_step_dft
                ));
            } else {
                let cost = match d.eval.readbacks {
                    Some(r) => format!("{} SCF iterations, {r} readbacks", d.eval.scf_steps),
                    None => format!("{} SCF iterations", d.eval.scf_steps),
                };
                ui.monospace(format!("{cost} · {} ms", d.eval.solve_ms));
            }
            ui.end_row();
            if let Some(a) = input.model().and_then(|m| m.last_audit.as_ref()).filter(|a| a.asynchronous) {
                ui.label("audit lag").on_hover_text("answered steps between the last audit's dispatch and its verdict (Model tab)");
                ui.monospace(format!("{} steps", a.unaudited_steps));
                ui.end_row();
            }
        });
    });
    ui.horizontal(|ui| {
        theme::section(ui, "E total and T");
        if ui.add(egui::Button::new(RichText::new("Graph tab ›").size(10.5).color(theme::ACCENT_INK)).frame(false)).clicked() {
            st.dock_tab = super::super::DockTab::Graph;
        }
    });
    sparkline(ui, &d.history, &d.history_source, &d.history_guess, 46.0, SPARK_WINDOW, GraphSeries::Both, false, false, st.temp_units);
    super::group(ui, st, "d-health", |ui, _| {
        theme::section(ui, "Health");
        match box_warning(&s.scene) {
            Some((c, w, _)) => health(ui, c, &w),
            None => health(ui, GREEN, &format!("all nuclei ≥ {BOX_MARGIN_WARN} Bohr from the box face")),
        }
        match &d.dt_warning {
            Some(w) => health(ui, AMBER, w),
            None => health(ui, GREEN, &format!("dt {:.3} fs (the warning starts above {DT_WARN_FS} fs)", d.dt_fs())),
        }
        for w in &d.eval.warnings {
            health(ui, AMBER, w);
        }
        for n in d.notes.iter().rev().take(3) {
            health(ui, theme::INK_FAINT, n);
        }
        if d.notes.len() > 3 {
            theme::note(ui, "older notes are in the Log tab");
        }
    });
    breakdown(ui, st, s);
}

fn health(ui: &mut Ui, c: Color32, text: &str) {
    ui.horizontal_wrapped(|ui| {
        theme::light(ui, c);
        let t = if c == GREEN || c == theme::INK_FAINT { RichText::new(text).color(theme::INK_DIM) } else { RichText::new(text).color(c) };
        ui.label(t.size(11.5));
    });
}

/// The latest evaluation's energy breakdown and solver diagnostics (collapsible, as before).
fn breakdown(ui: &mut Ui, st: &mut UiState, s: &crate::sim::Snapshot) {
    let d = &s.dynamics;
    let e = &d.eval.energy;
    let units = st.energy_units;
    let r = egui::CollapsingHeader::new("Energy breakdown and solver")
        .id_salt("energy-breakdown")
        .default_open(false)
        .open(search::opens(st, "d-breakdown"))
        .show(ui, |ui| {
            super::wide_grid(ui, "energy", 3).striped(true).show(ui, |ui| {
                let mut row = |name: &str, x: f64| {
                    ui.label(name);
                    energy_cells(ui, x, units, false);
                    ui.end_row();
                };
                row(pseudo::energy_label(&d.eval.pseudised), e.total);
                row("kinetic (electrons)", e.kinetic);
                row("external", e.external);
                row("Hartree", e.hartree);
                row("xc", e.xc);
                row("nuclear", e.nuclear);
            });
            let sigma = s.scene.sigma.unwrap_or(s.scene.grid_h);
            if d.eval.pseudised.is_empty() {
                ui.weak(format!("smeared nuclei (σ = {sigma} Bohr): absolute energies depend on σ and h"));
            } else {
                ui.weak(format!("{} — all-electron species are smeared nuclei (σ = {sigma} Bohr)", pseudo::VALENCE_NOTE));
            }
            if d.source == Source::Surrogate {
                ui.weak("surrogate step: only the total is the model's (it has no electrons); the parts above are 0");
            } else if d.source == Source::Fallback {
                ui.weak("fallback step: the DFT teacher's breakdown (with a trusted model E_KS above is the model's: total − teacher gap)");
            }
            if s.engine == Backend::Gpu {
                ui.weak("GPU (f32 grid): energies scatter by up to ~1e-5 Ha between solve paths");
            }
            super::wide_grid(ui, "scf", 2).striped(true).show(ui, |ui| {
                let mut row = |name: &str, v: String| {
                    ui.label(name);
                    ui.monospace(v);
                    ui.end_row();
                };
                row("SCF iterations", d.eval.scf_steps.to_string());
                // the worst LOBPCG residual among the solves the converged state rests on (the
                // GPU's deliberately loosened early SCF steps excluded)
                row("eigen residual (final)", format!("{:.1e}", d.eval.max_eigen_residual));
                row("Poisson residual", format!("{:.1e}", d.eval.poisson_residual));
                row(
                    "solve",
                    match d.eval.readbacks {
                        Some(r) => format!("{} ms, {r} readbacks ({})", d.eval.solve_ms, s.engine.label()),
                        None => format!("{} ms ({})", d.eval.solve_ms, s.engine.label()),
                    },
                );
                row("grid", format!("{}³ × {} Bohr", s.scene.grid_n, s.scene.grid_h));
                if let Some(max) = d.display.as_ref().map(|x| match x {
                    sim::Display::Cpu { max, .. } | sim::Display::Gpu { max, .. } => *max,
                }) {
                    row("ρ max", format!("{max:.3} e/Bohr³"));
                }
            });
            for n in &d.eval.notes {
                ui.weak(n);
            }
        });
    search::mark(ui, st, "d-breakdown", r.header_response.rect);
}
