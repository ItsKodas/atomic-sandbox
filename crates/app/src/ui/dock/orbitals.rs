//! The Orbitals tab (orbital-viewer spec §3, §3.1; plan Task 7): the `show` selector, the list of
//! Kohn–Sham orbitals — both spin channels, grouped by channel and in energy order — with ε in
//! Hartree and eV, occupation and tags, the picker (by track: the selection follows the orbital as
//! levels reorder), the per-mode controls (phase or |ψ|², "show unoccupied: m", the iso sliders,
//! the colour blend and scale, the include checkboxes) and the spin readout block.
//!
//! **Rule 7 lives here.** Every fact §2 of the spec lists is a constant below, and a tooltip or a
//! line of the list carries it: a Kohn–Sham orbital is an auxiliary object, not a measured
//! wavefunction ([`KS_NOTE`]); spin is intrinsic, orbitals are stationary ([`SPIN_NOTE`]); only the
//! HOMO eigenvalue means anything and in LSDA it is too shallow ([`HOMO_TIP`]); unoccupied
//! orbitals are eigenstates of the ground-state potential and the Kohn–Sham gap is not the
//! fundamental gap ([`GAP_TIP`]); a state of the box is not a state of the molecule
//! ([`box_tag`]); the smeared nucleus inverts shells ([`LONE_INVERTED`]); rotated and grid-split
//! sets are labelled with their spread ([`ROTATED_TIP`], [`grid_split_tag`]); the sign of an
//! orbital is arbitrary ([`SIGN_TIP`]). The list is built by pure functions of the payload
//! ([`RowView`], [`Levels`], [`unoccupied_lines`], [`spin_lines`]) so the tests read them without a
//! window. Energies are converted at this edge through `units` (rule 3); the payload speaks Hartree.
use super::super::{theme, Action, PanelInput, ShowMode, UiState, UNOCCUPIED_MAX};
use egui::{Color32, RichText, Ui};
use sim::{LoneAtomLabels, OrbitalDisplay, OrbitalInfo, OrbitalRequest, Spin, SpinSummary, UnoccupiedStatus};

// ---- the honesty text (spec §2; the plan pins these substrings) ----

/// §2 item 1: what an orbital is. In every mode's tooltip and the tab's header.
pub const KS_NOTE: &str = "Kohn–Sham orbital: an auxiliary object of DFT, not a measured electron wavefunction. The total density is the physical observable.";
/// §1 and §2: spin is intrinsic and nothing rotates.
pub const SPIN_NOTE: &str = "Spin is intrinsic (↑/↓), not a rotation. Orbitals are stationary: they move only with the nuclei.";
/// §2 item 6: the colours of a phase isosurface.
pub const SIGN_TIP: &str = "the sign of an orbital is arbitrary; only the relative sign of its lobes means anything";
/// §2 item 7: a rotated set.
pub const ROTATED_TIP: &str = "rotated within a degenerate set for display: any rotation of a degenerate set is an equally good eigenbasis, so the display rotates each set to a steady orientation (a lone atom's p lobes along the cell axes; in a molecule, toward the previous frame). Each member shown is an eigenfunction to within half the set's eigenvalue spread";
/// §2 item 3: the Kohn–Sham gap.
pub const GAP_TIP: &str = "the Kohn–Sham gap is not the fundamental gap (IP − EA): the Kohn–Sham potential has no derivative discontinuity in LSDA, so this gap is typically much smaller. Unoccupied orbitals are eigenstates of the ground-state potential, not excited states: their energies are not excitation energies";
/// §2 item 2: the HOMO.
pub const HOMO_TIP: &str = "highest occupied Kohn–Sham eigenvalue: the only one with a physical meaning (−ε_HOMO is the ionisation energy in exact DFT). In LSDA it is far too shallow because of self-interaction — the H atom's ε_1s↑ sits well above −0.5 Ha. The other eigenvalues are not excitation energies";
/// The unoccupied rows' header.
pub const UNOCCUPIED_TIP: &str = "unoccupied orbitals are eigenstates of the converged ground-state potential, computed for display (not excited states). LSDA's potential decays too fast, so many are not bound: in a finite box an unbound state becomes a state of the box. Their atomic names are never given. The Kohn–Sham gap is not the fundamental gap";
/// §5.7: a state of the box.
pub const BOX_STATE_TIP: &str = "a state of the simulation box, not of the molecule: its energy and shape come from the finite box (Dirichlet walls) and change with the box size";
/// §2 item 5: the smeared nucleus.
pub const SMEARED_TIP: &str = "smeared nucleus: every nucleus is a Gaussian of width σ = h, so a 1s orbital is wider and shallower than the physical one";
/// §2 item 5: the inversion of shells.
pub const LONE_INVERTED: &str = "shell order inverted by the smeared nucleus — not physical (needs a finer grid or pseudopotentials)";
/// Defined here, used by the M7 core rows (plan Task 8): a frozen core drawn from the generator's atom.
#[allow(dead_code)] // Task 8 (M7-gated) is the first user; the tooltip test pins the string now
pub const CORE_TAG: &str = "core: reconstructed for display";
/// §2 item 7: a lone atom's shell is split by the grid, not by physics.
pub const GRID_SPLIT_TIP: &str = "a lone atom's shell is degenerate in physics; the grid splits it by sampling the smeared nucleus at a position-dependent offset (the egg-box effect), by an amount that depends on where the atom sits relative to the grid. The members shown are eigenfunctions to within half this";

/// The tooltip of a `show` mode's button: what it draws, then the facts of §2 that apply to it.
pub fn mode_tip(mode: ShowMode) -> String {
    match mode {
        ShowMode::Density => "the total electron density ρ = ρ↑ + ρ↓ — the physical observable. Needs no orbitals".into(),
        ShowMode::Orbitals => format!("one orbital, as a ± phase surface or volume, or its square ψ̃². {KS_NOTE} {SIGN_TIP}. Needs a DFT step"),
        ShowMode::Coloured => format!("every included orbital in its own colour (the inner shells and the outer lobes together). {KS_NOTE} Each orbital is scaled to its own peak unless 'physical' is chosen. Needs a DFT step"),
        ShowMode::Spin => format!("the spin density m = ρ↑ − ρ↓ (amber ↑ excess, teal ↓ excess). {SPIN_NOTE} {KS_NOTE} Needs a DFT step"),
    }
}

// ---- number formats (the UI edge of rule 3) ----

/// `x` with `decimals` places and a true minus sign ("−0.500").
pub fn minus(x: f64, decimals: usize) -> String {
    format!("{x:.decimals$}").replace('-', "−")
}

/// ε in Hartree: "−0.500".
pub fn energy_ha(e: f64) -> String {
    minus(e, 3)
}

/// ε in eV, converted through `units`: "−13.61".
pub fn energy_ev(e: f64) -> String {
    minus(units::hartree_to_ev(e), 2)
}

fn arrow(s: Spin) -> &'static str {
    match s {
        Spin::Up => "↑",
        Spin::Down => "↓",
    }
}

// ---- the tags ----

/// One tag of a row: its text, the tooltip that explains it, and whether it warns (amber).
#[derive(Clone, Debug, PartialEq)]
pub struct Tag {
    pub text: String,
    pub tip: String,
    pub warn: bool,
}

fn tag(text: impl Into<String>, tip: impl Into<String>, warn: bool) -> Tag {
    Tag { text: text.into(), tip: tip.into(), warn }
}

/// "split by the grid by 0.50 mHa" (Ha → mHa).
pub fn grid_split_tag(spread: f64) -> Tag {
    tag(format!("split by the grid by {:.2} mHa", spread * 1e3), GRID_SPLIT_TIP, false)
}

/// The box-state tag, worded by the sign of ε (Task 4 review M2, rule 7): ε ≥ 0 is unbound in
/// LSDA at this box; ε < 0 flagged by the weight or the face rule is bound but its tail reaches
/// the walls.
pub fn box_tag(eps_max: f64) -> Tag {
    let text = if eps_max >= 0.0 {
        "state of the box — unbound in LSDA at this box"
    } else {
        "bound, but its tail reaches the box walls: energy box-affected"
    };
    tag(text, BOX_STATE_TIP, true)
}

/// Two levels closer than this are one (`qm::orbitals::TAU_MIN`, 1e-4 Ha: the display's smallest
/// degeneracy window): the ↑ and ↓ members of a closed shell differ by the solver's spin noise
/// (≤ 3e-6 Ha) and are both the HOMO.
const SAME_LEVEL: f64 = 1e-4;

/// HOMO and LUMO over both channels, read off the **eigenvalues** (`OrbitalInfo::eps_range`; for
/// a rotated set its lowest and highest), never off `OrbitalInfo::energy`, which for a rotated
/// row is the Rayleigh quotient of the displayed combination.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Levels {
    pub homo: Option<f64>,
    pub lumo: Option<f64>,
}

impl Levels {
    pub fn of(rows: &[OrbitalInfo]) -> Levels {
        let homo = rows.iter().filter(|r| r.occupation > 0.0).map(|r| r.eps_range.1).reduce(f64::max);
        let lumo = rows.iter().filter(|r| r.occupation == 0.0).map(|r| r.eps_range.0).reduce(f64::min);
        Levels { homo, lumo }
    }

    /// ε_LUMO − ε_HOMO, when both exist.
    pub fn gap(&self) -> Option<f64> {
        Some(self.lumo? - self.homo?)
    }

    /// Whether the row's set holds the HOMO: within [`SAME_LEVEL`] of it, so a closed shell's ↑
    /// and ↓ orbitals (equal to the solver's spin noise, ≤ 3e-6 Ha) are both the HOMO.
    fn is_homo(&self, r: &OrbitalInfo) -> bool {
        r.occupation > 0.0 && self.homo.is_some_and(|h| r.eps_range.1 >= h - SAME_LEVEL)
    }

    fn is_lumo(&self, r: &OrbitalInfo) -> bool {
        r.occupation == 0.0 && self.lumo.is_some_and(|l| r.eps_range.0 <= l + SAME_LEVEL)
    }
}

/// "HOMO −0.500 Ha (−13.61 eV) · LUMO … · Kohn–Sham gap …" (the tooltip is [`GAP_TIP`]).
pub fn levels_line(l: &Levels) -> Option<String> {
    let at = |e: f64| format!("{} Ha ({} eV)", energy_ha(e), energy_ev(e));
    match (l.homo, l.lumo, l.gap()) {
        (Some(h), Some(u), Some(g)) => Some(format!("HOMO {} · LUMO {} · Kohn–Sham gap {} Ha ({} eV)", at(h), at(u), minus(g, 3), minus(units::hartree_to_ev(g), 2))),
        (Some(h), _, _) => Some(format!("HOMO {}", at(h))),
        _ => None,
    }
}

/// What the list shows for one orbital.
#[derive(Clone, Debug, PartialEq)]
pub struct RowView {
    pub spin: Spin,
    pub track: u32,
    pub slot: u8,
    pub label: String,
    pub ha: String,
    pub ev: String,
    pub occupation: String,
    pub occupied: bool,
    pub tags: Vec<Tag>,
}

impl RowView {
    pub fn of(r: &OrbitalInfo, levels: &Levels) -> RowView {
        let mut tags = Vec::new();
        if levels.is_homo(r) {
            tags.push(tag("HOMO", HOMO_TIP, false));
        }
        if levels.is_lumo(r) {
            tags.push(tag("LUMO", GAP_TIP, false));
        }
        if let Some((d, s)) = r.cluster {
            tags.push(tag(format!("degenerate ×{d} (spread {:.2} mHa)", s * 1e3), "the eigenvalues of this set agree to within their spread; at the SCF tolerance they are one level", false));
        }
        if r.rotated {
            // the residual of the set's eigensolve, next to the spread (Task 4 review M4): the shown
            // combination's own residual can reach that plus half the spread
            let mut tip = ROTATED_TIP.to_string();
            let text = match r.residual {
                Some(res) => {
                    tip += &format!(". The eigensolve residual of the set is {res:.1e} Ha; the shown combination's own residual can reach it plus half the spread");
                    format!("rotated for display (eigensolve residual of the set {res:.1e})")
                }
                None => "rotated for display".to_string(),
            };
            tags.push(tag(text, tip, false));
        } else if let (Some(res), false) = (r.residual, r.not_converged) {
            tags.push(tag(format!("eigensolve residual of the set {res:.1e}"), "the largest ‖(H − ε)φ‖ among the members of this set: how well the display eigensolve converged", false));
        }
        if let Some(s) = r.grid_split {
            tags.push(grid_split_tag(s));
        }
        if r.box_state {
            tags.push(box_tag(r.eps_range.1));
        }
        if r.not_converged {
            let res = r.residual.map_or(String::new(), |x| format!(" (residual {x:.1e})"));
            tags.push(tag(format!("not converged{res}"), "the display eigensolve stopped at its iteration cap above its tolerance: the energy and shape are approximate. It is retried from these vectors when the worker is idle", true));
        }
        if r.smeared_1s {
            tags.push(tag("smeared nucleus: wider than the physical 1s", SMEARED_TIP, false));
        }
        RowView {
            spin: r.spin,
            track: r.track,
            slot: r.slot,
            label: r.label.to_string(),
            ha: energy_ha(r.energy),
            ev: energy_ev(r.energy),
            occupation: if (r.occupation - r.occupation.round()).abs() < 1e-9 { format!("{}", r.occupation.round() as i64) } else { format!("{:.2}", r.occupation) },
            occupied: r.occupation > 0.0,
            tags,
        }
    }

    /// The label's tooltip: what is on screen when this orbital is picked.
    pub fn label_tip(&self) -> String {
        format!("{KS_NOTE} Click to show it (the selection follows this orbital as the levels reorder). {SIGN_TIP}.")
    }
}

/// The lone-atom verdict as a line under the list (`None` for molecules and for a labelled atom
/// whose shells keep their order).
pub fn lone_atom_line(l: LoneAtomLabels) -> Option<String> {
    match l {
        LoneAtomLabels::NotLoneAtom | LoneAtomLabels::Labelled { shell_order_inverted: false } => None,
        LoneAtomLabels::Labelled { shell_order_inverted: true } => Some(LONE_INVERTED.into()),
        LoneAtomLabels::Unlabelled(why) => Some(format!("the orbitals read #k, not 1s/2p…: {why}")),
    }
}

/// The unoccupied rows' state lines (spec §4.5, plan Task 7), for a view asking `m` per channel
/// (`None` when no unoccupied orbitals are wanted): "computing unoccupied orbitals…" while the
/// payload on screen does not answer the request yet; "m requested, k shown" per channel when a
/// degenerate set cut by the block edge is hidden; "interrupted — recomputed when idle"; "showing
/// the m = held set; requested requested"; and the provider's own notes, minus the block-edge note
/// this function words itself.
pub fn unoccupied_lines(m: u8, status: Option<&UnoccupiedStatus>) -> Vec<String> {
    if m == 0 {
        return Vec::new();
    }
    let mut v = Vec::new();
    let Some(u) = status.filter(|u| u.requested == m.min(UNOCCUPIED_MAX)) else {
        v.push("computing unoccupied orbitals… (a display eigensolve: seconds at the default grid)".into());
        if let Some(u) = status {
            v.push(format!("showing the earlier request's rows (m = {}) meanwhile", u.requested));
        }
        return v;
    };
    for (ch, s) in [Spin::Up, Spin::Down].into_iter().enumerate() {
        if u.hidden[ch] > 0 {
            v.push(format!("{}: {} requested, {} shown: a degenerate set cut by the block edge is hidden", arrow(s), u.requested, u.shown[ch]));
        }
    }
    if u.interrupted {
        v.push("interrupted — recomputed when idle".into());
    }
    if u.held != 0 && u.held != u.requested {
        v.push(format!("showing the m = {} set; {} requested", u.held, u.requested));
    }
    for n in &u.notes {
        // the block-edge note is worded above from the counts
        if n.contains(" requested, ") && n.contains(" shown") {
            continue;
        }
        v.push(n.clone());
    }
    v
}

/// The spin readout block (spec §3, §5.5): n↑, n↓, ∫m, ∫|m| and max|m|/max ρ, and what they mean.
pub fn spin_lines(s: &SpinSummary) -> Vec<String> {
    let mut v = vec![
        format!("n↑ = {}, n↓ = {}", s.n_up, s.n_dn),
        format!("∫m = {:.4} (= n↑ − n↓ = {})", s.integral, s.n_up as i64 - s.n_dn as i64),
        format!("∫|m| = {:.4}", s.abs_integral),
    ];
    let ratio = if s.rho_max > 0.0 { s.max_abs / s.rho_max } else { 0.0 };
    // the display scale is floored at SPIN_FLOOR·max ρ: below it the field is f32 and eigensolver noise
    if s.scale > s.max_abs {
        v.push(format!("no net spin (max|m| = {ratio:.1e} · max ρ): a closed shell's m is numerical noise, and the view scales it by a floor instead of painting it at full colour"));
    } else {
        v.push(format!("max|m| / max ρ = {ratio:.3}"));
    }
    if s.abs_integral - s.integral.abs() > 1e-3 {
        v.push("∫|m| > |∫m|: spin of both signs is present — LSDA core polarisation, or a broken-symmetry singlet (a real feature of the functional)".into());
    }
    v
}

/// The sRGB swatch of a display colour (linear RGB in [0, 1], what the shaders write into the
/// sRGB frame).
pub fn swatch(c: [f32; 3]) -> Color32 {
    let enc = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        let y = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (y * 255.0).round() as u8
    };
    Color32::from_rgb(enc(c[0]), enc(c[1]), enc(c[2]))
}

/// Whether the include checkbox of a row is on (default: every occupied orbital).
pub fn included(st: &UiState, r: &OrbitalInfo) -> bool {
    match &st.included {
        None => r.occupation > 0.0,
        Some(set) => set.contains(&r.track),
    }
}

// ---- the tab ----

/// The `show` selector: one button per mode, probed as `{prefix}-{word}`; returns the picked mode.
pub fn show_selector(ui: &mut Ui, current: ShowMode, prefix: &str) -> Option<ShowMode> {
    let mut picked = None;
    egui::Frame::new()
        .fill(theme::APP)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(egui::CornerRadius::same(7))
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.horizontal_wrapped(|ui| {
                for mode in ShowMode::ALL {
                    let r = ui.add(egui::Button::selectable(current == mode, mode.title())).on_hover_text(mode_tip(mode));
                    super::super::probe(&format!("{prefix}-{}", mode.word()), &r);
                    if r.clicked() {
                        picked = Some(mode);
                    }
                }
            });
        });
    picked
}

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let payload: Option<&OrbitalDisplay> = input.snapshot.and_then(|s| s.dynamics.orbitals.as_ref());
    super::group(ui, st, "d-orb-mode", |ui, st| {
        ui.horizontal_wrapped(|ui| {
            ui.label("show");
            theme::info(ui, format!("{KS_NOTE} {SPIN_NOTE}"));
        });
        if let Some(m) = show_selector(ui, st.show, "show") {
            acts.push(Action::Show(m));
        }
        ui.add(egui::Label::new(RichText::new(KS_NOTE).size(11.0).color(theme::INK_DIM)).wrap());
        if st.show == ShowMode::Spin {
            ui.add(egui::Label::new(RichText::new(SPIN_NOTE).size(11.0).color(theme::INK_DIM)).wrap());
        }
        if let Some(l) = &input.orbital_label {
            ui.add(egui::Label::new(RichText::new(l).color(super::super::AMBER)).wrap());
        }
    });
    if st.show == ShowMode::Density {
        theme::note(ui, "pick Orbitals, All orbitals coloured or Spin density above (or View ▸ Density ▸ show) to see the Kohn–Sham orbitals; they need a DFT step.");
    }
    controls(ui, st, payload, acts);
    super::group(ui, st, "d-orb-spin", |ui, st| {
        theme::section(ui, "Spin readout");
        match (st.show == ShowMode::Spin, payload) {
            (true, Some(o)) => {
                for l in spin_lines(&o.spin) {
                    ui.add(egui::Label::new(RichText::new(l).size(11.5)).wrap());
                }
            }
            (true, None) => {
                theme::note(ui, "needs a DFT step");
            }
            (false, _) => {
                theme::note(ui, "n↑, n↓, ∫m, ∫|m| and max|m|/max ρ appear here in Spin density mode");
            }
        }
    });
    if let Some(o) = payload {
        for n in &o.notes {
            ui.add(egui::Label::new(RichText::new(n).size(11.5).color(super::super::AMBER)).wrap());
        }
    }
    list(ui, st, payload, acts);
}

/// The per-mode controls, always drawn (a search result can reveal them) and enabled for the modes
/// that use them: phase or |ψ|², the iso sliders, the colour blend and scale, the include reset,
/// and "show unoccupied".
fn controls(ui: &mut Ui, st: &mut UiState, payload: Option<&OrbitalDisplay>, acts: &mut Vec<Action>) {
    super::group(ui, st, "d-orb-style", |ui, st| {
        ui.add_enabled_ui(st.show == ShowMode::Orbitals, |ui| {
            ui.horizontal(|ui| {
                ui.label("draw as");
                if let Some(p) = theme::segmented(ui, st.orbital_phase, &[(true, "phase ±", SIGN_TIP), (false, "|ψ|²", "the orbital squared: where the electron in this orbital is likely to be, with the density's colour ramp")]) {
                    acts.push(Action::OrbitalPhase(p));
                }
            });
        });
    });
    super::group(ui, st, "d-orb-iso", |ui, st| {
        let spin = st.show == ShowMode::Spin;
        let signed = spin || (st.show == ShowMode::Orbitals && st.orbital_phase);
        let iso_on = st.volume_mode == super::super::VOLUME_ISO;
        let iso = if spin { &mut st.spin_iso } else { &mut st.phase_iso };
        ui.add_enabled(signed && iso_on, egui::Slider::new(iso, 0.02..=0.9).logarithmic(true).text(if spin { "iso (× the spin scale)" } else { "iso (× max|ψ|)" }))
            .on_hover_text("the ± isosurface (View ▸ Density ▸ render as isosurface, or the Isosurface switch in the view): surfaces at ±iso times the field's scale. For an orbital the scale is the largest |ψ| over its degenerate set");
    });
    super::group(ui, st, "d-orb-colour", |ui, st| {
        ui.add_enabled_ui(st.show == ShowMode::Coloured, |ui| {
            ui.horizontal(|ui| {
                ui.label("blend");
                if let Some(a) = theme::segmented(ui, st.colour_additive, &[(true, "additive", "emission only: inner shells shine through the outer ones"), (false, "weighted", "emission–absorption: the front shell hides what is behind it")]) {
                    acts.push(Action::ColourBlend { additive: a });
                }
            });
            ui.horizontal(|ui| {
                ui.label("scale");
                if let Some(p) = theme::segmented(ui, st.colour_physical, &[(false, "each to its peak", "each orbital scaled to its own peak: every shell equally visible"), (true, "physical", "scaled by the total density's maximum: summed over the orbitals the opacity is the density, and the hue is which orbital dominates locally")]) {
                    acts.push(Action::ColourScale { physical: p });
                }
            });
            let note = if st.colour_physical { "physical scale: hue = which orbital dominates; opacity = the density" } else { "each orbital scaled to its own peak" };
            theme::note(ui, note);
            if st.included.is_some() && super::super::probed("include-reset", ui.small_button("include every occupied orbital")).clicked() {
                acts.push(Action::IncludeAllOccupied);
            }
        });
    });
    super::group(ui, st, "d-orb-unocc", |ui, st| {
        ui.add_enabled_ui(matches!(st.show, ShowMode::Orbitals | ShowMode::Coloured), |ui| {
            let mut m = st.unoccupied;
            let r = ui.add(egui::Slider::new(&mut m, 0..=UNOCCUPIED_MAX).text("show unoccupied (m per spin)")).on_hover_text(UNOCCUPIED_TIP);
            super::super::probe("unoccupied", &r);
            // one request when the slider is let go (or clicked), not one per step of a drag
            if m != st.unoccupied && (r.drag_stopped() || !r.dragged()) {
                acts.push(Action::Unoccupied(m));
            }
            let status = payload.and_then(|o| o.unoccupied.as_ref());
            // while a change is pending the slider shows the new value; the lines speak of the request
            for l in unoccupied_lines(st.orbital_request().unoccupied, status) {
                ui.add(egui::Label::new(RichText::new(l).size(11.5).color(theme::INK_DIM)).wrap());
            }
        });
    });
}

/// The list: a heading per channel, a line per orbital and its tags, then the notes.
fn list(ui: &mut Ui, st: &mut UiState, payload: Option<&OrbitalDisplay>, acts: &mut Vec<Action>) {
    super::group(ui, st, "d-orb-list", |ui, st| {
        ui.horizontal_wrapped(|ui| {
            theme::section(ui, "Orbitals");
            theme::info(ui, format!("label · spin · ε (Ha) · ε (eV) · occupation · tags. {KS_NOTE} {HOMO_TIP}"));
        });
        let Some(o) = payload else {
            theme::note(ui, "needs a DFT step");
            return;
        };
        let levels = Levels::of(&o.rows);
        if let Some(l) = levels_line(&levels) {
            ui.add(egui::Label::new(RichText::new(l).monospace().size(11.0)).wrap()).on_hover_text(format!("{HOMO_TIP}. {GAP_TIP}"));
        }
        if o.rows.is_empty() {
            theme::note(ui, "no electrons — nothing to show");
        }
        let picked = match &o.what {
            OrbitalRequest::Orbital { spin, track, .. } => Some((*spin, *track)),
            _ => None,
        };
        for spin in [Spin::Up, Spin::Down] {
            let mut rows: Vec<&OrbitalInfo> = o.rows.iter().filter(|r| r.spin == spin).collect();
            rows.sort_by_key(|r| r.index);
            let n_occ = rows.iter().filter(|r| r.occupation > 0.0).count();
            if n_occ == 0 {
                ui.label(RichText::new(format!("{}: none occupied", arrow(spin))).color(theme::INK_DIM));
            } else {
                ui.label(RichText::new(format!("{} spin · {n_occ} occupied", arrow(spin))).strong());
            }
            let mut shown_unocc = false;
            for r in rows {
                if r.occupation == 0.0 && !shown_unocc {
                    shown_unocc = true;
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(format!("{} unoccupied", arrow(spin))).color(theme::INK_DIM));
                        theme::info(ui, UNOCCUPIED_TIP);
                    });
                }
                row(ui, st, r, &levels, picked == Some((r.spin, r.track)), acts);
            }
        }
        if let Some(l) = lone_atom_line(o.lone_atom) {
            ui.add(egui::Label::new(RichText::new(l).size(11.5).color(super::super::AMBER)).wrap());
        }
        if o.rows.iter().any(|r| r.occupation == 0.0) {
            ui.add(egui::Label::new(RichText::new(GAP_TIP).size(11.0).color(theme::INK_FAINT)).wrap());
        }
    });
}

/// One orbital: swatch, include box (all-coloured mode), the label as the picker, ε, occupation,
/// then its tags.
fn row(ui: &mut Ui, st: &mut UiState, r: &OrbitalInfo, levels: &Levels, picked: bool, acts: &mut Vec<Action>) {
    let v = RowView::of(r, levels);
    let colour = providers::orbitals::ORBITAL_PALETTE[r.slot as usize % providers::orbitals::ORBITAL_PALETTE.len()];
    ui.horizontal(|ui| {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 3.0, swatch(colour));
        resp.on_hover_text(if st.show == ShowMode::Coloured { "this orbital's colour in All orbitals coloured (a ↓ orbital shares its ↑ partner's colour)" } else { "this orbital's colour in All orbitals coloured" });
        if st.show == ShowMode::Coloured {
            let mut on = included(st, r);
            let cb = ui.checkbox(&mut on, "").on_hover_text("include this orbital in the colour volume");
            super::super::probe(&format!("include-{}", r.track), &cb);
            if cb.changed() {
                acts.push(Action::Include { track: r.track, on });
            }
        }
        let name = RichText::new(format!("{} {}", v.label, arrow(r.spin))).monospace();
        let b = ui.add(egui::Button::selectable(picked, name));
        let b = b.on_hover_text(v.label_tip());
        super::super::probe(&format!("orbital-{}-{}", arrow(r.spin), r.track), &b);
        if b.clicked() {
            acts.push(Action::PickOrbital { spin: r.spin, track: r.track });
        }
        ui.label(RichText::new(&v.ha).monospace()).on_hover_text("ε in Hartree: a Kohn–Sham eigenvalue; only the HOMO's has a physical meaning, and in LSDA it is too shallow");
        ui.label(RichText::new(&v.ev).monospace().color(theme::INK_DIM)).on_hover_text("ε in eV (1 Ha = 27.2114 eV; converted for display only)");
        ui.label(RichText::new(format!("occ {}", v.occupation)).monospace().color(theme::INK_DIM)).on_hover_text("occupation: 1 or 0 (integer aufbau)");
    });
    if !v.tags.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.add_space(18.0);
            for t in &v.tags {
                let c = if t.warn { super::super::AMBER } else { theme::INK_DIM };
                ui.label(RichText::new(&t.text).size(11.0).color(c)).on_hover_text(&t.tip);
            }
        });
    }
}
