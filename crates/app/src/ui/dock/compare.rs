//! Readouts ▸ Compare with reference (`crate::reference`): pick the method, press Compare; ours vs
//! PySCF's on the comparable quantities only — per-atom forces and their max deviation, the dipole,
//! and with "relax both" the bond lengths and angles of both relaxed geometries. Raw total energies
//! are never a deviation, and under pseudopotentials they are not shown at all (ours exclude the
//! cores). Nothing here feeds the dynamics.
use super::super::{search, theme, Action, PanelInput, UiState, AMBER, RED};
use crate::reference::{self, Method, Probe, RelaxPhase};
use egui::{RichText, Ui};

/// "O1", "H2", … (1-based, scene order).
fn atom(z: &[u8], i: usize) -> String {
    format!("{}{}", units::elements::symbol(z[i]), i + 1)
}

fn ang(x: f64) -> String {
    format!("{:.4}", units::bohr_to_angstrom(x))
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let open = if std::mem::take(&mut st.compare.open_request) { Some(true) } else { search::opens(st, "d-compare") };
    let r = egui::CollapsingHeader::new("Compare with reference (PySCF)").id_salt("compare-reference").default_open(false).open(open).show(ui, |ui| {
        st.compare.ensure_probe();
        body(ui, st, input, acts);
    });
    search::mark(ui, st, "d-compare", r.header_response.rect);
}

fn body(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    theme::note(ui, "An outside code answers the current geometry in its own process, for comparison only: it never drives the dynamics.");
    match &st.compare.probe {
        Probe::NotRun | Probe::Running(_) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("looking for Python with PySCF…").color(theme::INK_DIM));
            });
            return;
        }
        Probe::Missing(why) => {
            ui.horizontal_wrapped(|ui| {
                theme::light(ui, AMBER);
                ui.label(RichText::new(reference::INSTALL_HINT).color(AMBER));
            });
            theme::note(
                ui,
                "PySCF has no Windows build: install it inside WSL (`wsl -e python3` is tried), or set ATOMIC_SANDBOX_PYTHON to an interpreter that has it. Relaxing the reference also needs `pip install geometric`.",
            );
            let why = why.clone();
            egui::CollapsingHeader::new("what was tried").id_salt("compare-tried").show(ui, |ui| {
                ui.label(RichText::new(why).monospace().size(10.5).color(theme::INK_FAINT));
            });
            if ui.button("Check again").clicked() {
                st.compare.probe = Probe::NotRun;
            }
            return;
        }
        Probe::Ready(i) => {
            theme::note(ui, format!("PySCF {} · Python {} · `{}`{}", i.pyscf, i.python, reference::command_line(&i.cmd), if i.geomopt.is_none() { " · no optimiser (pip install geometric)" } else { "" }));
        }
    }
    let geomopt = st.compare.interpreter().is_some_and(|i| i.geomopt.is_some());
    let busy = st.compare.busy();
    ui.add_enabled_ui(!busy, |ui| {
        ui.horizontal(|ui| {
            ui.label("method");
            let before = st.compare.method;
            egui::ComboBox::from_id_salt("compare-method").selected_text(before.short()).show_ui(ui, |ui| {
                for m in Method::ALL {
                    ui.selectable_value(&mut st.compare.method, m, m.label());
                }
            });
            if st.compare.method != before {
                st.compare.timeout_min = st.compare.method.default_timeout_min();
            }
            ui.label(RichText::new(format!("/ {}", reference::DEFAULT_BASIS)).color(theme::INK_DIM));
        });
        ui.horizontal(|ui| {
            ui.add_enabled(geomopt, egui::Checkbox::new(&mut st.compare.relax, "relax both")).on_hover_text(
                "relax ours (FIRE, as Relax does — it moves the scene) and theirs (geomeTRIC) from this geometry, then compare the bond lengths and angles",
            );
            ui.label("timeout");
            ui.add(egui::DragValue::new(&mut st.compare.timeout_min).range(1..=600).suffix(" min"));
        });
    });
    if !geomopt {
        st.compare.relax = false;
    }
    ui.horizontal(|ui| {
        if busy {
            if ui.button("Cancel").on_hover_text("kill the reference process (our relax, if any, keeps running: Stop it in the toolbar)").clicked() {
                acts.push(Action::CancelCompare);
            }
            if let Some(job) = st.compare.run.as_ref().and_then(|r| r.job.as_ref()) {
                ui.spinner();
                ui.label(RichText::new(format!("PySCF running {:.0} s (timeout {} min)", job.started.elapsed().as_secs_f64(), job.timeout.as_secs() / 60)).color(theme::INK_DIM));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
            } else {
                ui.spinner();
                ui.label(RichText::new("waiting for our relax").color(theme::INK_DIM));
            }
        } else if ui.button("Compare").on_hover_text("answer the current geometry with PySCF (a separate process) and compare").clicked() {
            acts.push(Action::CompareReference { method: st.compare.method, relax: st.compare.relax });
        }
    });
    results(ui, st, input);
}

fn results(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    let Some(run) = st.compare.run.as_ref() else { return };
    let o = &run.ours;
    let m = run.request.method;
    ui.separator();
    theme::section(ui, &format!("PySCF {} / {} vs ours", m.short(), run.request.basis));
    theme::note(
        ui,
        format!(
            "ours: LSDA{}, {}, {}³ × {} Bohr, σ = {} Bohr{}",
            if o.estimated { " (the estimated solver's model)" } else { "" },
            o.engine,
            o.grid.0,
            o.grid.1,
            o.sigma,
            if o.pseudised { ", pseudopotentials" } else { ", all-electron" }
        ),
    );
    let answer = match &run.answer {
        None => {
            theme::note(ui, "waiting for PySCF…");
            None
        }
        Some(Err(e)) => {
            ui.label(RichText::new(format!("PySCF failed: {e}")).color(RED));
            None
        }
        Some(Ok(a)) => Some(a),
    };
    let still = input.snapshot.is_some_and(|s| s.dynamics.scene.nuclei.len() == o.z.len() && s.dynamics.scene.nuclei.iter().zip(&o.positions).all(|(n, p)| n.pos == *p));
    if !still && run.relax.is_none() {
        theme::note(ui, "the scene has moved since: these numbers are for the geometry Compare was pressed at");
    }
    if let Some(a) = answer {
        // energies: side by side only all-electron, never a deviation
        if o.pseudised {
            ui.horizontal_wrapped(|ui| {
                theme::light(ui, theme::INK_FAINT);
                ui.label(RichText::new("total energies not compared: our E_KS is the valence energy under pseudopotentials — the cores are excluded").size(11.5).color(theme::INK_DIM));
            });
        } else {
            super::wide_grid(ui, "compare-energy", 2).show(ui, |ui| {
                ui.label("E ours");
                ui.monospace(format!("{:.6} Ha", o.energy));
                ui.end_row();
                ui.label("E theirs");
                ui.monospace(format!("{:.6} Ha", a.energy));
                ui.end_row();
            });
            theme::note(ui, format!("not a deviation: our absolute energy carries the nuclear smearing (σ = {} Bohr) and the grid; theirs the basis{}", o.sigma, if m == Method::Lda { "" } else { " and another method" }));
        }
        forces(ui, o, a);
        dipole(ui, o, a, run.answer.is_some());
        if run.request.optimise {
            geometry(ui, run, a);
        }
        if !a.notes.is_empty() {
            theme::note(ui, format!("PySCF {}: {} · {:.1} s", a.pyscf, a.notes.join("; "), a.wall_s));
        }
        if a.forces.is_some() {
            ui.checkbox(&mut st.compare.arrows, "force arrows in the view (orange theirs, blue ours)").on_hover_text(
                "each atom's force at the compared geometry, PySCF's orange beside ours blue on one scale, drawn while the atoms are still there (display only)",
            );
        }
    } else if run.request.optimise {
        geometry_waiting(ui, run);
    }
}

fn forces(ui: &mut Ui, o: &reference::Ours, a: &reference::Answer) {
    let Some(theirs) = &a.forces else {
        theme::note(ui, "no reference forces for this method here (see the notes below)");
        return;
    };
    let Some(dev) = reference::force_deviation(&o.forces, theirs) else { return };
    theme::section(ui, "Forces (Ha/Bohr)");
    super::wide_grid(ui, "compare-forces", 4).striped(true).show(ui, |ui| {
        for h in ["atom", "|F| ours", "|F| theirs", "|ΔF|"] {
            ui.label(RichText::new(h).size(10.5).color(theme::INK_FAINT));
        }
        ui.end_row();
        for i in 0..o.z.len() {
            ui.label(atom(&o.z, i));
            ui.monospace(format!("{:.4}", reference::norm(o.forces[i])));
            ui.monospace(format!("{:.4}", reference::norm(theirs[i])));
            let c = if i == dev.worst && dev.max > 0.0 { AMBER } else { theme::INK };
            ui.monospace(RichText::new(format!("{:.4}", dev.per_atom[i])).color(c));
            ui.end_row();
        }
    });
    ui.monospace(format!("max |ΔF| {:.4} ({}) · RMS {:.4}", dev.max, atom(&o.z, dev.worst), dev.rms));
    if o.estimated {
        theme::note(ui, "ours are the model's forces at this step");
    }
}

fn dipole(ui: &mut Ui, o: &reference::Ours, a: &reference::Answer, done: bool) {
    theme::section(ui, "Dipole (e·Bohr)");
    let row = |ui: &mut Ui, who: &str, d: [f64; 3]| {
        ui.label(who);
        ui.monospace(format!("({:+.4}, {:+.4}, {:+.4})", d[0], d[1], d[2]));
        ui.monospace(format!("|μ| {:.4} = {:.3} D", reference::norm(d), units::au_dipole_to_debye(reference::norm(d))));
        ui.end_row();
    };
    super::wide_grid(ui, "compare-dipole", 3).show(ui, |ui| {
        if let Some(d) = o.dipole {
            row(ui, "ours", d);
        }
        if let Some(d) = a.dipole {
            row(ui, "theirs", d);
        }
    });
    match (o.dipole, a.dipole) {
        (Some(x), Some(y)) => {
            let dd = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
            ui.monospace(format!("|Δμ| {:.4} e·Bohr ({:.3} D)", reference::norm(dd), units::au_dipole_to_debye(reference::norm(dd))));
        }
        (None, _) if done => {
            theme::note(
                ui,
                "ours: no density measured at this geometry yet — it comes with the bonds and charges after a DFT solve here (estimated solver: the view's Bonds now button)",
            );
        }
        _ => {}
    }
    if o.charge != 0 {
        theme::note(ui, "a charged scene's dipole depends on the origin: both are about the cell centre");
    }
    if o.pseudised {
        theme::note(ui, "ours: valence density + ion charges Z_v (a frozen core is a neutral charge on its nucleus)");
    }
}

fn geometry_waiting(ui: &mut Ui, run: &reference::Run) {
    match run.relax {
        Some(RelaxPhase::Waiting | RelaxPhase::Running) => {
            theme::note(ui, "relaxing ours (FIRE) and theirs…");
        }
        _ => {
            if let Some(n) = &run.ours.relax_note {
                theme::note(ui, n);
            }
        }
    };
}

fn geometry(ui: &mut Ui, run: &reference::Run, a: &reference::Answer) {
    let o = &run.ours;
    theme::section(ui, "Relaxed geometries (Å, degrees)");
    if let Some(n) = &o.relax_note {
        theme::note(ui, n);
    }
    match a.optimise_converged {
        Some(false) => {
            theme::note(ui, "theirs did not converge in 100 steps");
        }
        Some(true) => {
            theme::note(ui, "theirs converged");
        }
        None => {}
    }
    let (Some(ours), Some(theirs)) = (o.relaxed.as_ref(), a.optimised.as_ref()) else {
        if matches!(run.relax, Some(RelaxPhase::Waiting | RelaxPhase::Running)) {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("waiting for our relax").color(theme::INK_DIM));
            });
        }
        return;
    };
    let lo = reference::bond_lengths(ours, &o.bonds);
    let lt = reference::bond_lengths(theirs, &o.bonds);
    super::wide_grid(ui, "compare-bonds", 4).striped(true).show(ui, |ui| {
        for h in ["bond", "ours", "theirs", "Δ"] {
            ui.label(RichText::new(h).size(10.5).color(theme::INK_FAINT));
        }
        ui.end_row();
        for (k, &(i, j)) in o.bonds.iter().enumerate() {
            ui.label(format!("{}–{}", atom(&o.z, i), atom(&o.z, j)));
            ui.monospace(ang(lo[k]));
            ui.monospace(ang(lt[k]));
            ui.monospace(format!("{:+.4}", units::bohr_to_angstrom(lo[k] - lt[k])));
            ui.end_row();
        }
    });
    let tri = reference::angle_triples(o.z.len(), &o.bonds);
    if !tri.is_empty() {
        super::wide_grid(ui, "compare-angles", 4).striped(true).show(ui, |ui| {
            for h in ["angle", "ours", "theirs", "Δ"] {
                ui.label(RichText::new(h).size(10.5).color(theme::INK_FAINT));
            }
            ui.end_row();
            for &t in &tri {
                let (x, y) = (reference::angle(ours, t), reference::angle(theirs, t));
                ui.label(format!("{}–{}–{}", atom(&o.z, t.0), atom(&o.z, t.1), atom(&o.z, t.2)));
                ui.monospace(format!("{x:.2}"));
                ui.monospace(format!("{y:.2}"));
                ui.monospace(format!("{:+.2}", x - y));
                ui.end_row();
            }
        });
    }
    theme::note(
        ui,
        if o.bonds_from_graph { "bonds: our bond graph's pairs" } else { "no bond graph: every pair is listed" },
    );
}
