//! Ctrl K: a fuzzy search over every control's name (the toolbar's, the menus' and the dock's,
//! collapsibles included). Picking a result reveals the control: it opens the menu or the dock
//! tab (and the collapsible) that holds it, scrolls it into view and outlines it for a moment.
use super::{theme, Action, DockTab, UiState};
use egui::{Align2, Key, RichText, Vec2};
use std::time::{Duration, Instant};

/// Where a control lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Toolbar,
    Menu(super::menus::MenuId),
    Dock(DockTab),
}

/// One searchable control: its name (and other words it answers to), where it lives, and the
/// anchor its widget group marks (`reveal`).
pub struct Entry {
    pub name: &'static str,
    pub also: &'static str,
    pub place: Place,
    pub anchor: &'static str,
}

use super::menus::MenuId as M;
use DockTab as D;
use Place::{Dock, Menu, Toolbar};

const fn e(name: &'static str, also: &'static str, place: Place, anchor: &'static str) -> Entry {
    Entry { name, also, place, anchor }
}

/// Every control of the chrome by name (Appendix A of the mockup, grouped by anchor).
pub const ENTRIES: &[Entry] = &[
    // toolbar
    e("Play / Pause", "space run start stop", Toolbar, "tb-time"),
    e("Step", "advance one time step", Toolbar, "tb-time"),
    e("Rewind", "reverse velocities v to −v", Toolbar, "tb-time"),
    e("Speed (steps per tick)", "slider fast slow", Toolbar, "tb-speed"),
    e("Watchable speed", "preset vibration", Toolbar, "tb-speed"),
    e("Measured fs/s", "rate simulated femtoseconds per second tick", Toolbar, "tb-speed"),
    e("Time step dt", "fs femtoseconds", Toolbar, "tb-dt"),
    e("Tool: Orbit", "camera rotate pan zoom", Toolbar, "tb-tool"),
    e("Tool: Place", "add atom nucleus drop", Toolbar, "tb-tool"),
    e("Tool: Select", "pick drag move nucleus", Toolbar, "tb-tool"),
    e("Element to place", "H He Li Be B C N O F Ne hydrogen oxygen carbon", Toolbar, "tb-elements"),
    e("Periodic table", "element picker search symbol name Z greyed pseudopotentials", Toolbar, "tb-elements"),
    e("Isotope to place", "mass number deuterium tritium D T ²H ³H ¹³C ¹⁸O heavy", Toolbar, "tb-elements"),
    e("Relax / Stop relax", "FIRE minimise optimise geometry relaxation", Toolbar, "tb-time"),
    e("Pin selected", "hold still pinned", Toolbar, "tb-selection"),
    e("Focus selected", "centre camera", Toolbar, "tb-selection"),
    e("Delete selected", "remove nucleus atom", Toolbar, "tb-selection"),
    e("Solver: CPU / GPU / Estimated", "backend surrogate model reference", Toolbar, "tb-solver"),
    e("Trust light", "trusted model fallback", Toolbar, "tb-solver"),
    // File
    e("Preset", "empty blank h h2 he lih h2o water hydrogen scene", Menu(M::File), "m-presets"),
    e("Reset scene", "reload preset", Menu(M::File), "m-presets"),
    e("Scene file path", "file name", Menu(M::File), "m-scene-file"),
    e("Save scene", "write file", Menu(M::File), "m-scene-file"),
    e("Load scene", "open read file", Menu(M::File), "m-scene-file"),
    e("Save model", "estimated surrogate .model", Menu(M::File), "m-model-file"),
    e("Load model", "estimated surrogate .model", Menu(M::File), "m-model-file"),
    e("Forget model", "drop samples estimated", Menu(M::File), "m-model-file"),
    // Scene
    e("Charge", "ion anion cation electrons", Menu(M::Scene), "m-electrons"),
    e("Multiplicity", "spin auto", Menu(M::Scene), "m-electrons"),
    e("σ nuclear smearing", "sigma auto = h", Menu(M::Scene), "m-electrons"),
    e("Quality (Fast / Balanced / Accurate / Precise / Custom)", "accuracy speed precision grid spacing h resolution slow fast", Menu(M::Scene), "m-quality"),
    e("Grid n", "points size resolution", Menu(M::Scene), "m-grid"),
    e("Grid spacing h", "Bohr resolution", Menu(M::Scene), "m-grid"),
    e("Apply grid", "cost box", Menu(M::Scene), "m-grid"),
    e("Container walls", "box piston size half-width", Menu(M::Scene), "m-container"),
    // View
    e("Density", "show what the view shows total electron density default", Menu(M::View), "m-density"),
    e("Orbitals", "show orbital one Kohn-Sham phase lobe psi p s d HOMO LUMO", Menu(M::View), "m-density"),
    e("All orbitals coloured", "show every orbital colour shells inner spheres", Menu(M::View), "m-density"),
    e("Spin density", "show up down magnetisation rho up minus rho down m amber teal", Menu(M::View), "m-density"),
    e("Colour dots by density / orbital", "dots probability samples colour", Menu(M::View), "m-density"),
    e("Density: volume / isosurface", "render cloud dots", Menu(M::View), "m-density"),
    e("Gamma", "density look", Menu(M::View), "m-density"),
    e("Opacity", "density look", Menu(M::View), "m-density"),
    e("Iso level (× ρ max)", "isosurface threshold", Menu(M::View), "m-density"),
    e("Refresh cloud every", "live density estimated", Menu(M::View), "m-density"),
    e("Atomic superposition between DFT clouds", "cloud estimated density", Menu(M::View), "m-density"),
    e("Cell box", "simulation box display", Menu(M::View), "m-display"),
    e("Floor grid", "display", Menu(M::View), "m-display"),
    e("Axes", "xyz display gizmo", Menu(M::View), "m-display"),
    e("Placement plane", "place grid blue", Menu(M::View), "m-display"),
    e("Grid spacing 1 Bohr / 1 Å", "angstrom display", Menu(M::View), "m-display"),
    e("Smooth motion", "interpolate glide", Menu(M::View), "m-display"),
    e("Bonds", "bond lines density graph show hide", Menu(M::View), "m-bonds"),
    e("Molecule names", "labels formula name water show hide", Menu(M::View), "m-bonds"),
    e("Name style", "systematic stoichiometric dihydrogen monoxide IUPAC hydride oxidane formula only common scientific names", Menu(M::View), "m-bonds"),
    e("Composition badge on hover", "isotope protons neutrons force hover", Menu(M::View), "m-bonds"),
    // Physics
    e("Temperature units K / °C / °F", "celsius fahrenheit kelvin degrees temperature scale", Menu(M::Physics), "m-thermostat"),
    e("Thermostat (Bussi CSVR)", "temperature T τ tau coupling heat", Menu(M::Physics), "m-thermostat"),
    e("Uniform acceleration (gravity hook)", "field g 9.8 external", Menu(M::Physics), "m-field"),
    e("COM correction", "centre of mass drift", Menu(M::Physics), "m-com"),
    e("Time step dt (Physics menu)", "fs", Menu(M::Physics), "m-dt"),
    e("Drag pull (drag force)", "pull force strength wheel spring tug drag reset", Menu(M::Physics), "m-drag"),
    e("Draft mode", "guess instead of falling back unsafe estimated", Menu(M::Physics), "m-draft"),
    e("Relax geometry (FIRE)", "minimise optimise relaxation stop", Menu(M::Physics), "m-relax"),
    // Help
    e("Controls", "mouse keys shortcuts help", Menu(M::Help), "m-controls"),
    e("Physics, honestly", "LSDA smeared limits help", Menu(M::Help), "m-honest"),
    // Readouts
    e("Temperature T", "kelvin", Dock(D::Readouts), "d-tiles"),
    e("Drift", "energy conservation Ha/ps health", Dock(D::Readouts), "d-tiles"),
    e("KE (nuclei)", "kinetic energy", Dock(D::Readouts), "d-energies"),
    e("E_KS", "Kohn-Sham energy", Dock(D::Readouts), "d-energies"),
    e("E total", "energy", Dock(D::Readouts), "d-energies"),
    e("E conserved", "energy invariant", Dock(D::Readouts), "d-energies"),
    e("Work done by tools", "drag thermostat pins", Dock(D::Readouts), "d-energies"),
    e("Model refit / force mismatch work", "estimated booking", Dock(D::Readouts), "d-energies"),
    e("Walls (piston) work", "container booking", Dock(D::Readouts), "d-energies"),
    e("Energy units Ha / eV", "hartree electronvolt", Dock(D::Readouts), "d-energies"),
    e("Sim time and step", "t fs", Dock(D::Readouts), "d-time"),
    e("Steps last tick, ms per step", "cost", Dock(D::Readouts), "d-time"),
    e("Per step cost (SCF steps, readbacks)", "solve ms", Dock(D::Readouts), "d-time"),
    e("Box warning", "face edge health", Dock(D::Readouts), "d-health"),
    e("dt warning", "health", Dock(D::Readouts), "d-health"),
    e("Dynamics notes", "thermostat Maxwell-Boltzmann", Dock(D::Readouts), "d-health"),
    e("Energy breakdown and solver", "kinetic external Hartree xc nuclear SCF residual Poisson ρ max", Dock(D::Readouts), "d-breakdown"),
    // Graph
    e("Graph: E total and T", "sparkline plot history curve", Dock(D::Graph), "d-graph"),
    // Model
    e("This step (surrogate / fallback / DFT)", "source who answered", Dock(D::Model), "d-model"),
    e("Uncertainty", "bound model force std", Dock(D::Model), "d-model"),
    e("Trusted", "trust model", Dock(D::Model), "d-model"),
    e("Samples, model epoch, last fit, last refit", "model", Dock(D::Model), "d-model"),
    e("Fallbacks", "percent", Dock(D::Model), "d-model"),
    e("Teacher gap", "DFT − model", Dock(D::Model), "d-model"),
    e("Last audit", "audit lag", Dock(D::Model), "d-model"),
    e("Mismatch work", "model", Dock(D::Model), "d-model"),
    e("DFT audit every", "cadence audits", Dock(D::Model), "d-audits"),
    e("Sampling plan", "perturbations σ_r hot burst scan seed", Dock(D::Model), "d-plan"),
    e("Sampling plan preset", "quick look gentle standard hot reactive use case", Dock(D::Model), "d-plan"),
    e("Start sampling / Sample more", "pass teach", Dock(D::Model), "d-sample"),
    e("Cancel sampling", "pass", Dock(D::Model), "d-sample"),
    e("Model file (save / load / forget)", "estimated", Dock(D::Model), "d-model-file"),
    e("Draft mode (Model tab)", "guess unsafe", Dock(D::Model), "d-draft"),
    e("What this model is (and is not)", "honest estimated", Dock(D::Model), "d-model-about"),
    // Activity
    e("Jobs (what is running)", "activity progress cancel pause stop sampling refit audit solve work", Dock(D::Activity), "d-jobs"),
    e("Counters (samples, teacher solves)", "activity samples in the model teacher solves this session ticks", Dock(D::Activity), "d-counters"),
    e("Solve log", "activity teacher solves rule outcome fallback audit pass why reasons", Dock(D::Activity), "d-solve-log"),
    e("Trust checklist (why untrusted)", "activity conditions refused samples box audit residual classes", Dock(D::Activity), "d-trust"),
    e("Training views (rmse, parity, pair curves, coverage)", "activity plots model vs DFT uncertainty band histogram", Dock(D::Activity), "d-training"),
    e("Event timeline", "activity fallbacks audits refits quarantines passes", Dock(D::Activity), "d-timeline"),
    // Orbitals
    e("Orbital list (ε in Ha and eV, occupation, tags)", "Kohn-Sham eigenvalues HOMO LUMO gap degenerate box state", Dock(D::Orbitals), "d-orb-list"),
    e("Show unoccupied orbitals (m)", "virtual empty LUMO eigensolve slider", Dock(D::Orbitals), "d-orb-unocc"),
    e("Orbital: phase ± / |ψ|²", "sign red blue density squared style", Dock(D::Orbitals), "d-orb-style"),
    e("Orbital iso level (phase and spin)", "isosurface threshold lobes", Dock(D::Orbitals), "d-orb-iso"),
    e("Orbital colours: blend and scale", "additive weighted physical peak include", Dock(D::Orbitals), "d-orb-colour"),
    e("Spin readout", "n up n down integral m magnetisation noise", Dock(D::Orbitals), "d-orb-spin"),
    // Selection
    e("Selected nucleus", "position composition electrons", Dock(D::Selection), "d-selection"),
    e("Distance guides", "bond length Bohr Å", Dock(D::Selection), "d-guides"),
    e("Isotope of the selected nucleus", "mass number deuterium tritium swap", Dock(D::Selection), "d-selection"),
    e("Bonds of the selected nucleus", "bonded pair verdict density", Dock(D::Selection), "d-bonds"),
    e("Molecules (formula and name)", "labels water net charge", Dock(D::Selection), "d-molecules"),
    e("Show the atom card", "schematic Bohr", Dock(D::Selection), "d-card"),
    // Log
    e("Log", "errors refused notes status history messages", Dock(D::Log), "d-log"),
];

/// How long a revealed control stays outlined.
const REVEAL_FOR: Duration = Duration::from_millis(2500);

#[derive(Default)]
pub struct SearchState {
    pub open: bool,
    pub query: String,
    /// The highlighted result (arrow keys).
    pub cursor: usize,
    /// The anchor a pick revealed, when, and whether its widget has scrolled it into view.
    pub reveal: Option<(&'static str, Instant, bool)>,
    /// Set when the search opens, so the text field takes the focus.
    focus: bool,
}

impl SearchState {
    pub fn toggle(&mut self) {
        self.open = !self.open;
        if self.open {
            self.query.clear();
            self.cursor = 0;
            self.focus = true;
        }
    }

    /// The anchor being revealed now.
    pub fn revealing(&self) -> Option<&'static str> {
        self.reveal.filter(|(_, t, _)| t.elapsed() < REVEAL_FOR).map(|(a, _, _)| a)
    }
}

/// Subsequence match score (higher is better), `None` when `query` is not a subsequence of the
/// text: consecutive runs and word starts score, a match in the name beats one in `also`.
pub fn score(query: &str, entry: &Entry) -> Option<i32> {
    let q: Vec<char> = query.to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    if q.is_empty() {
        return Some(0);
    }
    let sub = |text: &str| -> Option<i32> {
        let t: Vec<char> = text.to_lowercase().chars().collect();
        let (mut qi, mut s, mut prev) = (0usize, 0i32, None::<usize>);
        for (i, c) in t.iter().enumerate() {
            if qi < q.len() && *c == q[qi] {
                s += 1;
                if prev == Some(i.wrapping_sub(1)) {
                    s += 3;
                }
                if i == 0 || !t[i - 1].is_alphanumeric() {
                    s += 2;
                }
                prev = Some(i);
                qi += 1;
            }
        }
        (qi == q.len()).then_some(s)
    };
    match (sub(entry.name), sub(entry.also)) {
        (Some(a), _) => Some(a + 10),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// The results for `query`, best first (at most `n`).
pub fn results(query: &str, n: usize) -> Vec<&'static Entry> {
    let mut v: Vec<(i32, usize, &Entry)> =
        ENTRIES.iter().enumerate().filter_map(|(i, e)| score(query, e).map(|s| (s, i, e))).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    v.into_iter().take(n).map(|(_, _, e)| e).collect()
}

/// Reveals `entry`: opens its menu or dock tab and starts the outline.
pub fn pick(ctx: &egui::Context, st: &mut UiState, entry: &'static Entry) {
    match entry.place {
        Place::Toolbar => {}
        Place::Menu(m) => egui::Popup::open_id(ctx, super::menus::popup_id(m)),
        Place::Dock(tab) => {
            st.dock_open = true;
            st.dock_tab = tab;
        }
    }
    st.search.reveal = Some((entry.anchor, Instant::now(), false));
    st.search.open = false;
}

#[cfg(test)]
thread_local! {
    /// Test hook: every anchor `mark` was called with (the anchor test in `ui::tests`).
    pub(crate) static MARKED: std::cell::RefCell<std::collections::HashSet<String>> = Default::default();
}

/// Outlines the widget group `rect` marked with `anchor` while it is being revealed, and scrolls
/// it into view once.
pub fn mark(ui: &egui::Ui, st: &mut UiState, anchor: &str, rect: egui::Rect) {
    #[cfg(test)]
    MARKED.with(|m| {
        m.borrow_mut().insert(anchor.to_string());
    });
    if st.search.revealing() != Some(anchor) {
        return;
    }
    let t = st.search.reveal.as_ref().map_or(0.0, |r| r.1.elapsed().as_secs_f32());
    let a = (1.0 - t / REVEAL_FOR.as_secs_f32()).clamp(0.0, 1.0);
    ui.painter().rect_stroke(
        rect.expand(3.0),
        6.0,
        egui::Stroke::new(2.0, theme::ACCENT.gamma_multiply(a)),
        egui::StrokeKind::Outside,
    );
    if let Some(r) = &mut st.search.reveal {
        if !r.2 {
            ui.scroll_to_rect(rect, Some(egui::Align::Center));
            r.2 = true;
        }
    }
    ui.ctx().request_repaint();
}

/// Whether a collapsible holding `anchor` should be forced open this frame.
pub fn opens(st: &UiState, anchor: &str) -> Option<bool> {
    (st.search.revealing() == Some(anchor)).then_some(true)
}

/// The search palette (top centre), while open.
pub fn show(ctx: &egui::Context, st: &mut UiState, _acts: &mut Vec<Action>) {
    if !st.search.open {
        return;
    }
    let found = results(&st.search.query, 12);
    let (mut up, mut down, enter, esc) = ctx.input(|i| {
        (i.key_pressed(Key::ArrowUp), i.key_pressed(Key::ArrowDown), i.key_pressed(Key::Enter), i.key_pressed(Key::Escape))
    });
    if found.is_empty() {
        (up, down) = (false, false);
    }
    if down {
        st.search.cursor = (st.search.cursor + 1).min(found.len().saturating_sub(1));
    }
    if up {
        st.search.cursor = st.search.cursor.saturating_sub(1);
    }
    let mut picked: Option<&'static Entry> = None;
    egui::Area::new(egui::Id::new("control-search"))
        .order(egui::Order::Foreground)
        .fade_in(false)
        .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 48.0))
        .show(ctx, |ui| {
            theme::card().fill(theme::PANEL).show(ui, |ui| {
                ui.set_width(460.0);
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut st.search.query)
                        .hint_text("Find a control or setting…")
                        .desired_width(f32::INFINITY),
                );
                if st.search.focus {
                    edit.request_focus();
                    st.search.focus = false;
                }
                if edit.changed() {
                    st.search.cursor = 0;
                }
                ui.add_space(4.0);
                if found.is_empty() {
                    theme::note(ui, "no control by that name");
                }
                for (k, entry) in found.iter().enumerate() {
                    let place = match entry.place {
                        Place::Toolbar => "toolbar".to_string(),
                        Place::Menu(m) => format!("menu › {}", m.title()),
                        // the Model tab shows its controls only with the estimated solver
                        Place::Dock(DockTab::Model) if st.backend != crate::sim::Backend::Estimated => {
                            format!("dock › {} (pick Estimated first)", DockTab::Model.title())
                        }
                        // the Activity tab shows its trust, training and timeline groups likewise
                        Place::Dock(DockTab::Activity) if st.backend != crate::sim::Backend::Estimated && matches!(entry.anchor, "d-trust" | "d-training" | "d-timeline") => {
                            format!("dock › {} (pick Estimated first)", DockTab::Activity.title())
                        }
                        Place::Dock(t) => format!("dock › {}", t.title()),
                    };
                    let r = ui.add(
                        egui::Button::selectable(
                            k == st.search.cursor,
                            egui::RichText::new(entry.name),
                        )
                        .right_text(RichText::new(place).size(11.0).color(theme::INK_FAINT))
                        .min_size(Vec2::new(ui.available_width(), 24.0)),
                    );
                    if r.clicked() {
                        picked = Some(*entry);
                    }
                }
                theme::note(ui, "Up / Down to choose · Enter to reveal · Esc to close");
            });
        });
    if enter {
        picked = picked.or_else(|| found.get(st.search.cursor).copied());
    }
    if let Some(e) = picked {
        pick(ctx, st, e);
    } else if esc {
        st.search.open = false;
    }
}
