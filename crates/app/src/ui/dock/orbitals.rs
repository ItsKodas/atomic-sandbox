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
//!
//! **Orbitals by atom** (orbital-atom-shares spec; user request 2026-11-12: "I want to see which
//! orbitals are mostly on that atom"). With the payload's Hirshfeld shares (`OrbitalInfo::shares`,
//! indexed by `OrbitalDisplay::share_z`, used only when that is the scene's own Z list) the list is
//! grouped by atom ([`groups`]: under an atom holding ≥ [`MAJORITY`], else under `shared · …`;
//! box states and rows of an earlier geometry under their own headings, with no share), each row
//! shows its largest share ([`share_text`]; the tooltip [`SHARE_TIP`] says it is a split of space,
//! not measured), and the atoms selected in the view light their rows ([`lit`]) while every other
//! row is drawn faint — never hidden, never re-sorted (the grouping reads the shares only). A
//! one-nucleus scene and a payload without usable shares list as before ([`grouped`]; the latter
//! with [`shares_note`]). Display only: nothing here is an `Action`.
use super::super::{theme, Action, PanelInput, ShowMode, UiState, UNOCCUPIED_MAX};
use super::selection::atom_name;
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
pub const HOMO_TIP: &str = "highest occupied Kohn–Sham eigenvalue: in exact DFT −ε_HOMO is the ionisation energy, but not here — in LSDA it is far too shallow because of self-interaction (the H atom's ε_1s↑ sits well above −0.5 Ha), and a smeared nucleus moves it further. The other eigenvalues are not excitation energies";
/// The HOMO tag of a lone atom whose shell order the smeared nucleus inverted.
pub const HOMO_INVERTED_TIP: &str = "shell order inverted: this level is not the physical HOMO — the smeared nucleus pushed the 2s above the 2p (needs a finer grid or pseudopotentials)";
/// Molecules with a heavy nucleus: the core orbitals are smeared too (spec §2 item 5).
pub const SMEARED_MOLECULE_TIP: &str = "smeared nuclei: every nucleus is a Gaussian of width σ = h, and for the heaviest one here Zσ is at least 0.5, so its lowest (core) orbitals are much wider and shallower than the physical ones, their eigenvalues are not the physical core levels, and the valence levels are shifted too";
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
/// A drawn orbital's boundary surface reaches the box walls (user request 2026-10-20).
pub const TOUCHES_BOX: &str = "boundary touches the box: enlarge the box";
/// Its tooltip.
pub const TOUCHES_BOX_TIP: &str = "the surface that encloses the chosen share of this orbital reaches the outer sixth of the simulation box, where its walls (ψ = 0 on them) bend the orbital: the box squeezes it, so its shape and energy are not the free atom's or molecule's. Enlarge the box (Scene ▸ grid), or lower the enclosed share to see whether only the outer tail is affected";
/// The tooltip of an unoccupied row's include box.
pub const UNOCC_INCLUDE_TIP: &str = "include this unoccupied orbital: drawn only as a boundary surface (draw: shell or filled), at full weight — it holds no electrons, so by density there is nothing to draw. An eigenstate of the ground-state potential, not an excited state";
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
    // below the display's resolution (0.005 mHa: a grid-symmetric site) "split by 0.00 mHa" would claim a split
    if spread * 1e3 < 0.005 {
        return tag("not split by the grid (< 0.005 mHa)", GRID_SPLIT_TIP, false);
    }
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
    /// A lone atom whose shell order the smeared nucleus inverted: its "HOMO" is not the physical one.
    pub inverted: bool,
    /// The LUMO row is a state of the box (its energy is the Dirichlet box's), is not converged, or
    /// the unoccupied rows are an earlier geometry's: the gap is then not a number to quote.
    pub lumo_box: bool,
    pub lumo_unconverged: bool,
    pub stale: bool,
    /// Some orbital is partly occupied (`0 < f < 1`: a degenerate shell shared equally under
    /// `Occupations::DegenerateShells`, M7 — an open shell of a pseudo-atom): there is no gap between
    /// partly filled degenerate orbitals, so none is quoted.
    pub fractional: bool,
}

impl Levels {
    pub fn of(rows: &[OrbitalInfo]) -> Levels {
        let homo = rows.iter().filter(|r| r.occupation > 0.0).map(|r| r.eps_range.1).reduce(f64::max);
        let lumo = rows.iter().filter(|r| r.occupation == 0.0).map(|r| r.eps_range.0).reduce(f64::min);
        let at_lumo = |r: &&OrbitalInfo| r.occupation == 0.0 && lumo.is_some_and(|l| r.eps_range.0 <= l + SAME_LEVEL);
        let (lumo_box, lumo_unconverged) = (rows.iter().filter(at_lumo).any(|r| r.box_state), rows.iter().filter(at_lumo).any(|r| r.not_converged));
        let fractional = rows.iter().any(|r| r.occupation > 1e-9 && r.occupation < 1.0 - 1e-9);
        Levels { homo, lumo, inverted: false, lumo_box, lumo_unconverged, stale: false, fractional }
    }

    /// With the unoccupied rows' status: rows that are not of the current geometry make the gap stale.
    pub fn with_unoccupied(mut self, u: Option<&UnoccupiedStatus>) -> Levels {
        self.stale = u.is_some_and(|u| !u.current);
        self
    }

    /// With the lone-atom verdict of the payload.
    pub fn with_lone_atom(mut self, l: LoneAtomLabels) -> Levels {
        self.inverted = matches!(l, LoneAtomLabels::Labelled { shell_order_inverted: true });
        self
    }

    /// ε_LUMO − ε_HOMO, when both exist.
    pub fn gap(&self) -> Option<f64> {
        if self.fractional {
            return None;
        }
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
    let line = match (l.homo, l.lumo, l.gap()) {
        (Some(h), Some(u), Some(g)) => format!("HOMO {} · LUMO {} · Kohn–Sham gap {} Ha ({} eV)", at(h), at(u), minus(g, 3), minus(units::hartree_to_ev(g), 2)),
        (Some(h), _, _) if l.fractional => format!("HOMO {} (partly filled shell: no HOMO–LUMO gap — the orbitals of a shared shell are one level)", at(h)),
        (Some(h), _, _) => format!("HOMO {}", at(h)),
        _ => return None,
    };
    // a gap that is the box's, unconverged or stale is not a property of the molecule (rule 7)
    let mut caveats: Vec<&str> = Vec::new();
    if l.lumo_box && l.gap().is_some() {
        caveats.push("the LUMO is a state of the box, so this gap depends on the box size and is not a property of the molecule");
    }
    if l.lumo_unconverged && l.gap().is_some() {
        caveats.push("the LUMO is not converged");
    }
    if l.stale && l.gap().is_some() {
        caveats.push("the unoccupied rows are from an earlier geometry");
    }
    let mut line = line;
    if !caveats.is_empty() {
        line = format!("{line} — {}", caveats.join("; "));
    }
    Some(if l.inverted { format!("{line} — shell order inverted: not the physical HOMO") } else { line })
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
            if levels.inverted {
                tags.push(tag("HOMO (shell order inverted: not the physical HOMO)", format!("{HOMO_INVERTED_TIP}. {HOMO_TIP}"), true));
            } else {
                tags.push(tag("HOMO", HOMO_TIP, false));
            }
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
            tags.push(tag(format!("not converged{res}"), "the display eigensolve stopped at its iteration cap above its tolerance: the energy and shape are approximate. It is recomputed from these vectors at the next paused step, edit or view change", true));
        }
        if r.smeared_1s {
            tags.push(tag("smeared nucleus: wider than the physical 1s", SMEARED_TIP, false));
        }
        if r.touches_box && r.occupation > 0.0 {
            tags.push(tag(TOUCHES_BOX, TOUCHES_BOX_TIP, true));
        }
        RowView {
            spin: r.spin,
            track: r.track,
            slot: r.slot,
            label: r.label.to_string(),
            ha: energy_ha(r.energy),
            ev: energy_ev(r.energy),
            occupation: occupation_text(r.occupation),
            occupied: r.occupation > 0.0,
            tags,
        }
    }

    /// One row for a closed shell's pair ([`paired`]; user request 2026-11-11): the ↑ member's ε
    /// and plain tags, the two occupations summed (H₂'s σ_g reads 2; a fractional shell sums its
    /// two fractions), and the warning tags of both members (final review M1: a caveat is never
    /// hidden — `not_converged` and the box tags come from each channel's own eigensolve, so the ↓
    /// member's can differ). A warning both members carry is shown once; one only a member carries
    /// is prefixed with its arrow ("↓ not converged (residual 3.0e-4)").
    pub fn pair(up: &OrbitalInfo, dn: &OrbitalInfo, levels: &Levels) -> RowView {
        let u = RowView::of(up, levels);
        let d = RowView::of(dn, levels);
        let has = |v: &RowView, t: &Tag| v.tags.iter().any(|x| x.warn && x.text == t.text);
        let marked = |t: &Tag, s: Spin| Tag { text: format!("{} {}", arrow(s), t.text), tip: format!("the {} orbital of this pair only: {}", arrow(s), t.tip), warn: true };
        // the ↑ member's tags in their order, then the warnings only the ↓ member carries
        let mut tags: Vec<Tag> = u.tags.iter().map(|t| if t.warn && !has(&d, t) { marked(t, Spin::Up) } else { t.clone() }).collect();
        tags.extend(d.tags.iter().filter(|t| t.warn && !has(&u, t)).map(|t| marked(t, Spin::Down)));
        RowView { occupation: occupation_text(up.occupation + dn.occupation), tags, ..u }
    }

    /// The label's tooltip: what is on screen when this orbital is picked.
    pub fn label_tip(&self) -> String {
        format!("{KS_NOTE} Click to show it (the selection follows this orbital as the levels reorder). {SIGN_TIP}.")
    }
}

/// An occupation as the list prints it: an integer bare, a fraction to two places.
fn occupation_text(f: f64) -> String {
    if (f - f.round()).abs() < 1e-9 { format!("{}", f.round() as i64) } else { format!("{f:.2}") }
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

/// What the dots are, for the mode on screen (rule 7): samples of where an electron is likely to be
/// only for the density and a single orbital's ψ̃²; in All orbitals coloured they sample the colour
/// volume's strength — each orbital scaled to its own peak (not the electron density), or with the
/// physical scale the density of the included orbitals — and for the spin density |ρ↑ − ρ↓|.
pub fn dots_note(st: &UiState) -> String {
    match st.show {
        ShowMode::Spin => "dots sample |ρ↑ − ρ↓|, coloured by sign: where the spins are unbalanced — not electrons".into(),
        ShowMode::Coloured if st.colour_look.enclose.is_some() => format!(
            "dots fill each orbital's boundary surface (the region holding {} % of its probability{}) evenly, in its colour — a picture of the orbital's extent, not the density, not electrons",
            st.colour_look.enclose.unwrap_or(90),
            if st.colour_look.shell { ", only a shell just inside it" } else { "" }
        ),
        ShowMode::Coloured if st.colour_physical && st.included.is_none() => "dots sample the density summed over the occupied orbitals, coloured by the dominant orbital — where an electron is likely to be, not electrons".into(),
        ShowMode::Coloured if st.colour_physical => "dots sample the density of the included orbitals, coloured by the dominant one — not the whole electron density, not electrons".into(),
        ShowMode::Coloured => "dots sample each orbital scaled to its own peak — not the electron density, not electrons".into(),
        _ => "dots are samples of where an electron is likely to be — not electrons".into(),
    }
}

/// The largest Z·σ (σ in Bohr: the scene's, else h) over the nuclei the solve smears — a pseudised species has no core
/// on the grid, so it is left out (water under `All`: no smeared nucleus at all; found in the M7 Task 13 Step 5 check).
pub fn largest_smeared_z_sigma(scene: &sim::Scene, pseudised: &[sim::PseudoSummary]) -> f64 {
    let sigma = scene.sigma.unwrap_or(scene.cell.h);
    scene.nuclei.iter().filter(|n| !pseudised.iter().any(|p| p.z == n.z)).map(|n| n.z as f64 * sigma).fold(0.0, f64::max)
}

/// A molecule's list-level note when its heaviest nucleus is smeared (Zσ ≥ `SMEARED_ZSIGMA`, spec
/// §2 item 5): its core orbitals are not the physical ones. `None` for a lone atom (its 1s row
/// carries the tag) and below the threshold. `z_sigma` is the largest Z·σ in the scene (σ in Bohr).
pub fn smeared_molecule_line(lone: LoneAtomLabels, z_sigma: f64) -> Option<String> {
    (lone == LoneAtomLabels::NotLoneAtom && z_sigma >= qm::orbitals::SMEARED_ZSIGMA)
        .then(|| format!("smeared nuclei: the heaviest nucleus here has Zσ = {z_sigma:.1} ≥ {}, so the lowest (core) orbitals are much wider and shallower than the physical ones and the valence levels are shifted too", qm::orbitals::SMEARED_ZSIGMA))
}

/// The notes of a payload the tab shows. A pick the viewer adopted from the HOMO placeholder
/// (`UiState::orbital_pick_adopted`) never was the user's selection, so the producer's "selected
/// orbital no longer exists" note — raised when its tracks restart, e.g. after an isolated-atom job
/// of the estimated solver — is untrue for it and is dropped; the viewer re-adopts the HOMO. The
/// note of a pick the user made stays.
pub fn visible_notes<'a>(o: &'a OrbitalDisplay, pick_adopted: bool) -> Vec<&'a String> {
    o.notes.iter().filter(|n| !(pick_adopted && n.starts_with(NO_LONGER_EXISTS))).collect()
}

/// The provider's note for a selection that vanished (`providers::orbitals::ViewState::pick`).
pub const NO_LONGER_EXISTS: &str = "selected orbital no longer exists";

/// The unoccupied rows' state lines (spec §4.5, plan Task 7), for a view asking `m` per channel
/// (`None` when no unoccupied orbitals are wanted): "computing unoccupied orbitals…" while the
/// payload on screen does not answer the request yet; "m requested, k shown" per channel when a
/// degenerate set cut by the block edge is hidden; "interrupted — recomputed when idle"; "showing
/// the m = held set; requested requested"; and the provider's own notes, minus the block-edge note
/// this function words itself.
pub fn unoccupied_lines(m: u8, status: Option<&UnoccupiedStatus>, waiting: bool) -> Vec<String> {
    if m == 0 {
        return Vec::new();
    }
    let mut v = Vec::new();
    // an eigensolve at the same m (a paused step, an edit, an estimated pick) is running while a
    // request is out and its payload has not arrived (`Viewer::orbital_waiting`)
    if waiting && status.is_some_and(|u| u.requested == m.min(UNOCCUPIED_MAX)) {
        v.push("computing unoccupied orbitals…".into());
    }
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
    if s.no_net_spin() && s.n_up + s.n_dn > 0 {
        // the provider sends no field: nothing is drawn, and its note (`providers::orbitals::NO_NET_SPIN`)
        // says so above; here the numbers behind it (rule 7)
        v.push(format!("m is within the solver's noise (a closed shell has m = 0): ∫|m| = {:.1e} e is inside the noise budget {:.1e} e (≈ 2 × the SCF tolerance + 2 × the electrons × the eigensolver tolerance + f32 roundoff — an empirical envelope, `SpinSummary::noise_l1`); max|m| = {ratio:.1e} · max ρ; not drawn", s.abs_integral, s.noise));
    } else if s.scale > s.max_abs {
        // the display scale is floored at SPIN_FLOOR·max ρ: below it the field is f32 and eigensolver noise
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

/// The rows the colour view draws for an include list (`None` = every occupied orbital) and look,
/// as the provider resolves them (`providers::orbitals::ViewState::pick`): an included unoccupied
/// row only as a boundary surface, and without the core rows (more than `sim::CORE_GAP` below the
/// highest occupied one) when they are hidden.
pub fn drawn_rows<'a>(rows: &'a [OrbitalInfo], include: Option<&[u32]>, look: &sim::ColourLook) -> Vec<&'a OrbitalInfo> {
    let homo = rows.iter().filter(|r| r.occupation > 0.0).map(|r| r.energy).fold(f64::NEG_INFINITY, f64::max);
    rows.iter()
        .filter(|r| match include {
            Some(tracks) => tracks.contains(&r.track) && (r.occupation > 0.0 || look.enclose.is_some()),
            None => r.occupation > 0.0,
        })
        .filter(|r| !look.hide_core || r.energy >= homo - sim::CORE_GAP)
        .collect()
}

/// The legend's warning when a drawn occupied orbital's boundary reaches the box walls, naming them
/// once per pair ("boundary touches the box: enlarge the box (#2, #3)"); unoccupied rows carry
/// their own box tag instead.
pub fn touching_line(drawn: &[&OrbitalInfo]) -> Option<String> {
    let mut names: Vec<String> = Vec::new();
    for r in drawn.iter().filter(|r| r.touches_box && r.occupation > 0.0) {
        let n = r.label.to_string();
        if !names.contains(&n) {
            names.push(n);
        }
    }
    (!names.is_empty()).then(|| format!("{TOUCHES_BOX} ({})", names.join(", ")))
}

/// Whether the include checkbox of a row is on (default: every occupied orbital).
pub fn included(st: &UiState, r: &OrbitalInfo) -> bool {
    match &st.included {
        None => r.occupation > 0.0,
        Some(set) => set.contains(&r.track),
    }
}

/// The paired heading's tooltip.
pub const PAIRED_TIP: &str = "each orbital holds one ↑ and one ↓ electron with the same shape and energy (a closed shell); the lists split into ↑ and ↓ as soon as the two spins differ by more than 0.1 mHa";

/// The rows of each channel, ↑ then ↓, in index (energy) order.
fn channels(rows: &[OrbitalInfo]) -> [Vec<&OrbitalInfo>; 2] {
    [Spin::Up, Spin::Down].map(|s| {
        let mut v: Vec<&OrbitalInfo> = rows.iter().filter(|r| r.spin == s).collect();
        v.sort_by_key(|r| r.index);
        v
    })
}

/// Whether the two channels are one closed shell, so the list folds them into one ↑↓ list (user
/// request 2026-11-11: "fold them in … when they're the exact same, less confusing"): the spin
/// readout says no net spin ([`SpinSummary::no_net_spin`]: ∫|m| inside the solver's noise, so a
/// broken-symmetry singlet — equal ε, different shapes — stays split), both channels hold the same
/// occupied rows, and every row both channels show, matched by index, has the same occupation and
/// ε within [`SAME_LEVEL`] (the solver's spin noise is ≤ 3e-6 Ha). An unoccupied row only one
/// channel shows still pairs (He with 4 unoccupied per spin: a degenerate set cut by the block
/// edge is hidden in one channel and not the other — user report 2026-11-12; with no net spin
/// both channels see the same potential); [`merged`] lists it on its own. A list without an
/// occupied row is not paired.
pub fn paired(rows: &[OrbitalInfo], spin: &SpinSummary) -> bool {
    let [up, dn] = channels(rows);
    let occupied = |c: &[&OrbitalInfo]| c.iter().filter(|r| r.occupation > 0.0).count();
    spin.no_net_spin()
        && occupied(&up) > 0
        && occupied(&up) == occupied(&dn)
        && merged(&up, &dn).iter().all(|m| match m {
            (u, Some(d)) => (u.occupation - d.occupation).abs() < 1e-9 && (u.energy - d.energy).abs() <= SAME_LEVEL,
            (r, None) => r.occupation == 0.0,
        })
}

/// The two channels' rows merged by index: a pair where both show the row, else the one row alone.
fn merged<'a>(up: &[&'a OrbitalInfo], dn: &[&'a OrbitalInfo]) -> Vec<(&'a OrbitalInfo, Option<&'a OrbitalInfo>)> {
    let mut out = Vec::with_capacity(up.len().max(dn.len()));
    let (mut i, mut j) = (0, 0);
    while i < up.len() || j < dn.len() {
        match (up.get(i), dn.get(j)) {
            (Some(u), Some(d)) if u.index == d.index => {
                out.push((*u, Some(*d)));
                i += 1;
                j += 1;
            }
            (Some(u), Some(d)) if u.index < d.index => {
                out.push((*u, None));
                i += 1;
            }
            (Some(u), None) => {
                out.push((*u, None));
                i += 1;
            }
            (_, Some(d)) => {
                out.push((*d, None));
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    out
}

// ---- where an orbital lives (orbital-atom-shares spec, goals 1–4; user request 2026-11-12) ----

/// An orbital is listed under an atom when that atom holds at least this share of it, and a
/// selection lights a row when the selected atoms together hold at least this much ([`lit`]) — spec,
/// "Calls made after the builder check": H₂'s σ_g (50/50 to the grid's noise) reads `shared` on
/// every step instead of flickering between its atoms. A display margin, not physics. Every
/// threshold here is judged at the whole percent the list prints (Task 4 review, Minor 2): a row
/// that reads "55 %" is never under shared, an atom whose hover reads "25 %" is named.
pub const MAJORITY: f64 = 0.55;
/// A `shared · …` heading names the atoms holding at least this share, largest first.
pub const SHARED_MIN: f64 = 0.25;
/// A row's tooltip lists the atoms holding at least this share, and a `shared · …` heading counts
/// the other atoms holding at least this much of one of its orbitals ("+ k others").
pub const HOVER_MIN: f64 = 0.01;
/// The share's tooltip (rule 7: the share is a choice of split, not a measurement).
pub const SHARE_TIP: &str = "the share of this orbital on each atom, s_A = ∫ w_A |ψ|²: a split of space between the atoms (Hirshfeld, like the partial charges), not something measured — another split (Mulliken, Bader) gives other numbers. An orbital is listed under an atom holding at least 55 % of it, else under shared (naming the atoms holding at least 25 % and counting the others holding at least 1 %); the 55 % line is a display margin, not physics, and every line is judged at the whole percent shown (“> 99 %”: it rounds to 100 %, but in a molecule the other atoms always hold some). A degenerate set shows its average (a single member's share depends on the display's rotation of the set); a paired ↑↓ row shows the ↑ orbital's";
/// The list's note while the free-atom references the shares are weighted by are being solved.
pub const SHARES_PENDING: &str = "atom shares need the atom references: being solved";
/// The prefix of the producer's note when the shares failed their Σ_A s = 1 check
/// (`providers::orbitals::settle_shares`).
const SHARES_OFF: &str = "orbital shares off";

/// Under whose heading a row is listed (spec, goal 2 and "Calls made after the builder check").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Owner {
    /// One atom (scene index) holds at least [`MAJORITY`] of it.
    Atom(usize),
    /// No atom does: the atoms holding at least [`SHARED_MIN`], largest first (equal at the whole
    /// percent shown: by atom number). Empty when the orbital is spread thinner than that.
    Shared(Vec<usize>),
    /// A state of the box (`OrbitalInfo::box_state`): no share (its far tails' split is the box's).
    Box,
    /// An unoccupied row solved at an earlier geometry (`UnoccupiedStatus::current` false).
    Earlier,
    /// A row without shares.
    Unknown,
}

/// One heading of the grouped list and its rows (indices into the rows given to [`groups`], in
/// their given order). `others`: for a [`Owner::Shared`] group, the atoms its heading does not
/// name that hold at least [`HOVER_MIN`] of one of its rows ("+ k others", ruling (f)); display
/// only — the group is keyed by `owner` alone, so a tail crossing 1 % never splits a group. 0 for
/// the other owners.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub owner: Owner,
    pub rows: Vec<usize>,
    pub others: usize,
}

/// A share as the list prints it: whole percent.
fn percent(s: f64) -> i64 {
    (100.0 * s).round() as i64
}

/// Whether `s` reaches the threshold `min` at the whole percent shown (review Minor 2).
fn at_least(s: f64, min: f64) -> bool {
    percent(s) >= percent(min)
}

/// A share as text: "92 %"; a share that rounds to 100 % reads "> 99 %" (review Minor 3: shares
/// are shown only in a molecule, where the other atoms always hold some of it, below the 1 % the
/// hover lists).
fn percent_text(s: f64) -> String {
    match percent(s) {
        p if p >= 100 => "> 99 %".into(),
        p => format!("{p} %"),
    }
}

/// The atoms by share, largest first; shares equal at the whole percent shown go by atom number,
/// so H₂'s 0.5000001 / 0.4999999 orders (and names) the same on every step.
fn ranked(shares: &[f64]) -> Vec<usize> {
    let mut k: Vec<usize> = (0..shares.len()).collect();
    k.sort_by_key(|&a| (std::cmp::Reverse(percent(shares[a])), a));
    k
}

/// The owner of a row with shares (no memory: the current shares only).
pub fn owner_of(shares: &[f64]) -> Owner {
    let order = ranked(shares);
    match order.first() {
        Some(&best) if at_least(shares[best], MAJORITY) => Owner::Atom(best),
        _ => Owner::Shared(order.into_iter().filter(|&k| at_least(shares[k], SHARED_MIN)).collect()),
    }
}

/// The list's groups (spec, goal 2): `rows` in the order they are drawn, `z` the species the
/// shares index (the scene's), `unocc_current` whether the unoccupied rows are of this geometry.
/// A row goes under [`Owner::Box`] if it is a state of the box, under [`Owner::Earlier`] if it is an
/// unoccupied row while those are an earlier geometry's, under [`Owner::Unknown`] without shares (or
/// shares not one per atom of `z`), else under [`owner_of`]. A degenerate set carries one averaged
/// share (the provider's), so it stays together. Groups go atoms by number, then shared groups by
/// their first atom (a group naming none last), then box states, earlier, unknown; rows keep their
/// order. A shared group counts its `others` over its rows. Selection plays no part: the list never
/// re-sorts as atoms are selected (anti-goal).
pub fn groups(rows: &[&OrbitalInfo], z: &[u8], unocc_current: bool) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let owner = if r.box_state {
            Owner::Box
        } else if !unocc_current && r.occupation == 0.0 {
            Owner::Earlier
        } else {
            match r.shares.as_deref() {
                Some(s) if !z.is_empty() && s.len() == z.len() => owner_of(s),
                _ => Owner::Unknown,
            }
        };
        match out.iter_mut().find(|g| g.owner == owner) {
            Some(g) => g.rows.push(i),
            None => out.push(Group { owner, rows: vec![i], others: 0 }),
        }
    }
    for g in out.iter_mut() {
        if let Owner::Shared(named) = &g.owner {
            let mut others: Vec<usize> = Vec::new();
            for s in g.rows.iter().filter_map(|&i| rows[i].shares.as_deref()) {
                others.extend((0..s.len()).filter(|&k| at_least(s[k], HOVER_MIN) && !named.contains(&k)));
            }
            others.sort_unstable();
            others.dedup();
            g.others = others.len();
        }
    }
    let rank = |o: &Owner| match o {
        Owner::Atom(k) => (0, *k, Vec::new()),
        Owner::Shared(v) => (1, v.first().copied().unwrap_or(usize::MAX), v.clone()),
        Owner::Box => (2, 0, Vec::new()),
        Owner::Earlier => (3, 0, Vec::new()),
        Owner::Unknown => (4, 0, Vec::new()),
    };
    out.sort_by_key(|g| rank(&g.owner));
    out
}

/// A group's heading: `O #0`, `shared · O #0–H #1–H #2`, `shared · C #0 + 4 others`, `shared ·
/// spread over 5 atoms`, `box states`, … (atom names as the Selection tab gives them; `others` is
/// [`Group::others`]). A shared heading never reads as if the named atoms held all of it (ruling
/// (f), rule 7).
pub fn heading(owner: &Owner, others: usize, z: &[u8]) -> String {
    let name = |k: usize| atom_name(z.get(k).copied().unwrap_or(0), k);
    match owner {
        Owner::Atom(k) => name(*k),
        Owner::Shared(v) if v.is_empty() => format!("shared · spread over {others} atoms"),
        Owner::Shared(v) => {
            let named = v.iter().map(|&k| name(k)).collect::<Vec<_>>().join("–");
            match others {
                0 => format!("shared · {named}"),
                1 => format!("shared · {named} + 1 other"),
                k => format!("shared · {named} + {k} others"),
            }
        }
        Owner::Box => "box states".into(),
        Owner::Earlier => "from an earlier geometry".into(),
        Owner::Unknown => "no atom shares".into(),
    }
}

/// A heading's tooltip.
fn heading_tip(owner: &Owner) -> String {
    match owner {
        Owner::Atom(_) => format!("orbitals with at least {:.0} % of their share on this atom. {SHARE_TIP}", 100.0 * MAJORITY),
        Owner::Shared(_) => format!(
            "orbitals no single atom holds {:.0} % of — bonds and delocalised orbitals — naming the atoms holding at least {:.0} %, largest first; \"+ k others\" counts the other atoms holding at least {:.0} % of one of them (\"spread over k atoms\" when none holds {:.0} %). {SHARE_TIP}",
            100.0 * MAJORITY,
            100.0 * SHARED_MIN,
            100.0 * HOVER_MIN,
            100.0 * SHARED_MIN
        ),
        Owner::Box => format!("no atom share: splitting a box state's far tails between the atoms would present an artefact as a fact. {BOX_STATE_TIP}"),
        Owner::Earlier => "unoccupied orbitals solved at an earlier geometry (the display eigensolve reruns when idle): shares belong to the orbitals they were computed with, so these get none".into(),
        Owner::Unknown => "no share was computed for these orbitals".into(),
    }
}

/// Whether a row stays at full strength under the selection (spec, goal 3): the selected atoms
/// together hold at least [`MAJORITY`] of it — a bond's shared orbital lights when both its atoms
/// are selected. Nothing selected: every row is lit.
pub fn lit(shares: &[f64], selected: &[usize]) -> bool {
    selected.is_empty() || at_least(selected.iter().filter_map(|&k| shares.get(k)).sum::<f64>(), MAJORITY)
}

/// The selected atoms: the view's selection and the Edit tool's group (`UiState::edit_group`),
/// each once, in scene order, within the scene's `n` nuclei.
pub fn selected_atoms(selected: Option<usize>, edit_group: &[usize], n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = selected.into_iter().chain(edit_group.iter().copied()).filter(|&k| k < n).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// A row's share as the list prints it: its largest, `H #0 50 %` (spec, goal 1).
pub fn share_text(shares: &[f64], z: &[u8]) -> String {
    let k = ranked(shares).first().copied().unwrap_or(0);
    format!("{} {}", atom_name(z.get(k).copied().unwrap_or(0), k), percent_text(shares.get(k).copied().unwrap_or(0.0)))
}

/// The share's hover: every atom holding at least [`HOVER_MIN`], largest first, then [`SHARE_TIP`].
pub fn share_hover(shares: &[f64], z: &[u8]) -> String {
    let atoms: Vec<String> = ranked(shares)
        .into_iter()
        .filter(|&k| at_least(shares[k], HOVER_MIN))
        .map(|k| format!("{} {}", atom_name(z.get(k).copied().unwrap_or(0), k), percent_text(shares[k])))
        .collect();
    format!("{}\n{SHARE_TIP}", atoms.join(" · "))
}

/// Whether the list is grouped by atom: more than one nucleus (spec, goal 4: a lone atom lists as
/// before) and shares indexing exactly the scene's species — a payload solved before an edit
/// (`share_z` another Z list) never pairs its shares with the wrong atoms (controller ruling, Task 4).
pub fn grouped(o: &OrbitalDisplay, scene_z: &[u8]) -> bool {
    scene_z.len() > 1 && o.share_z.as_deref() == Some(scene_z)
}

/// The list's one-line note when a molecule's payload has no usable shares (the list is then as
/// before): why — the payload is from before an edit, its shares failed their own check (the
/// producer's reason), the free-atom references are being solved (`refs_pending`), or they are
/// not in yet. `None` when grouped, for a lone atom, and without rows.
pub fn shares_note(o: &OrbitalDisplay, scene_z: &[u8], refs_pending: bool) -> Option<String> {
    if scene_z.len() <= 1 || o.rows.is_empty() || grouped(o, scene_z) {
        return None;
    }
    if o.share_z.is_some() {
        return Some("atom shares not shown: these orbitals were solved before the last edit, for other atoms".into());
    }
    if o.notes.iter().any(|n| n.starts_with(SHARES_OFF)) {
        // the producer's own note ("orbital shares off: Σ = …") is drawn above the list with the
        // payload's notes: said once (Task 4 review, Minor 4)
        return Some("atom shares not shown: they failed their own check (Σ over the atoms = 1; the note above)".into());
    }
    Some(if refs_pending { SHARES_PENDING.into() } else { "atom shares need the atom references: not in for these orbitals yet".into() })
}

/// What the list needs besides the payload: the scene's species, the selected atoms, whether the
/// free-atom references are being solved, and the smeared-core note's Zσ.
struct ListInput<'a> {
    z_sigma: f64,
    scene_z: &'a [u8],
    selected: &'a [usize],
    refs_pending: bool,
}

/// How one row is drawn besides its orbital: its share (text and hover; `None` = no share column)
/// and whether the selection dims it.
#[derive(Default)]
struct Place {
    share: Option<(String, String)>,
    faint: bool,
}

impl Place {
    /// For a row of `owner` with `shares` (the ↑ member's, for a pair) when the list is grouped.
    fn of(owner: &Owner, shares: Option<&[f64]>, z: &[u8], selected: &[usize]) -> Place {
        let shares = shares.filter(|_| matches!(owner, Owner::Atom(_) | Owner::Shared(_)));
        Place {
            share: shares.map(|s| (share_text(s, z), share_hover(s, z))),
            // a row without a share (box, earlier, unknown) is not on the selected atoms as far as
            // anything here knows: dimmed under a selection like any other row that is not
            faint: !shares.map_or(selected.is_empty(), |s| lit(s, selected)),
        }
    }
}

// ---- the tab ----

/// The `show` selector: one button per mode, probed as `{prefix}-{word}`; returns the picked mode.
/// "None" (first) hides the electron cloud altogether (`hidden`, `UiState::hide_cloud`); picking a
/// mode shows it again.
pub fn show_selector(ui: &mut Ui, current: ShowMode, hidden: &mut bool, prefix: &str) -> Option<ShowMode> {
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
                let r = ui.add(egui::Button::selectable(*hidden, "None")).on_hover_text("no electron cloud: only the atoms, their bonds and labels");
                super::super::probe(&format!("{prefix}-none"), &r);
                if r.clicked() {
                    *hidden = true;
                }
                for mode in ShowMode::ALL {
                    let r = ui.add(egui::Button::selectable(!*hidden && current == mode, mode.title())).on_hover_text(mode_tip(mode));
                    super::super::probe(&format!("{prefix}-{}", mode.word()), &r);
                    if r.clicked() {
                        *hidden = false;
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
        if let Some(m) = show_selector(ui, st.show, &mut st.hide_cloud, "show") {
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
        theme::note(ui, "pick Orbitals, All orbitals coloured or Spin density above (or View ▸ What you see) to see the Kohn–Sham orbitals; they need a DFT step.");
    }
    controls(ui, st, payload, input.orbital_waiting, acts);
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
        for n in visible_notes(o, st.orbital_pick_adopted) {
            ui.add(egui::Label::new(RichText::new(n).size(11.5).color(super::super::AMBER)).wrap());
        }
    }
    // the largest Zσ of the scene's all-electron nuclei: the molecule's smeared-core note
    let pseudised = super::super::pseudo::pseudised(input);
    let z_sigma = input.snapshot.map_or(0.0, |s| largest_smeared_z_sigma(&s.dynamics.scene, &pseudised));
    // the groups name and the selection lights the scene's atoms (orbital-atom-shares spec)
    let scene_z: Vec<u8> = input.scene.nuclei.iter().map(|n| n.z).collect();
    let selected = selected_atoms(input.selected, &st.edit_group, scene_z.len());
    let refs_pending = input.snapshot.is_some_and(|s| !s.dynamics.bonds_pending.is_empty());
    list(ui, st, payload, &ListInput { z_sigma, scene_z: &scene_z, selected: &selected, refs_pending }, acts);
}

/// The per-mode controls, always drawn (a search result can reveal them) and enabled for the modes
/// that use them: phase or |ψ|², the iso sliders, the colour blend and scale, the include reset,
/// and "show unoccupied".
fn controls(ui: &mut Ui, st: &mut UiState, payload: Option<&OrbitalDisplay>, waiting: bool, acts: &mut Vec<Action>) {
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
            .on_hover_text("the ± isosurface (View ▸ How it looks ▸ render as isosurface, or the Isosurface switch in the view): surfaces at ±iso times the field's scale. For an orbital the scale is the largest |ψ| over its degenerate set");
    });
    super::group(ui, st, "d-orb-colour", |ui, st| {
        ui.add_enabled_ui(st.show == ShowMode::Coloured, |ui| {
            // boundary surfaces (user request 2026-10-19): each orbital as the region enclosing a
            // share of its probability, a shell just inside it (the default) or filled; or the density
            let mut look = st.colour_look;
            ui.horizontal(|ui| {
                ui.label("draw");
                let now = match look.enclose {
                    None => 2,
                    Some(_) if look.shell => 0,
                    Some(_) => 1,
                };
                if let Some(m) = theme::segmented(ui, now, &[
                    (0, "shell", "each orbital's boundary surface: a thin shell just inside the surface that encloses the chosen share of its probability — the textbook picture, hollow inside"),
                    (1, "filled", "each orbital's whole region inside that surface, evenly filled"),
                    (2, "density", "each orbital by its density, fading out smoothly (no boundary)"),
                ]) {
                    let pct = look.enclose.unwrap_or(90);
                    (look.enclose, look.shell) = match m {
                        0 => (Some(pct), true),
                        1 => (Some(pct), false),
                        _ => (None, look.shell),
                    };
                }
            });
            if let Some(mut pct) = look.enclose {
                let r = ui.add(egui::Slider::new(&mut pct, 50..=99).suffix(" %").text("enclosed"))
                    .on_hover_text("the share of each orbital's probability inside its surface (textbooks use 90 %): higher draws bigger, fuzzier-edged lobes");
                // one request when the slider is let go (or clicked), not one per step of a drag
                if pct != look.enclose.unwrap_or(90) && (r.drag_stopped() || !r.dragged()) {
                    look.enclose = Some(pct);
                }
            }
            ui.checkbox(&mut look.hide_core, "hide core orbitals").on_hover_text(format!(
                "leave out the inner-shell orbitals (more than {} Ha below the highest occupied one, e.g. Li's 1s) so the valence lobes show; a pseudised atom has none",
                sim::CORE_GAP
            ));
            if look != st.colour_look {
                acts.push(Action::ColourLook(look));
            }
            // the presets only where they act (they would push the list down in the other modes)
            if st.show == ShowMode::Coloured {
                presets(ui, st, payload, acts);
            }
            ui.horizontal(|ui| {
                ui.label("blend");
                if let Some(a) = theme::segmented(ui, st.colour_additive, &[(true, "additive", "emission only: inner shells shine through the outer ones"), (false, "weighted", "emission–absorption: the front shell hides what is behind it")]) {
                    acts.push(Action::ColourBlend { additive: a });
                }
            });
            ui.add_enabled_ui(st.colour_look.enclose.is_none(), |ui| ui.horizontal(|ui| {
                ui.label("scale");
                if let Some(p) = theme::segmented(ui, st.colour_physical, &[(false, "each to its peak", "each orbital scaled to its own peak: every shell equally visible"), (true, "physical", "scaled by the total density's maximum: summed over the orbitals the opacity is the density, and the hue is which orbital dominates locally")]) {
                    acts.push(Action::ColourScale { physical: p });
                }
            }));
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
            // the name above the slider: beside it, it ran past a narrow panel's edge
            ui.label("show unoccupied (m per spin)").on_hover_text(UNOCCUPIED_TIP);
            let r = ui.add(egui::Slider::new(&mut m, 0..=UNOCCUPIED_MAX)).on_hover_text(UNOCCUPIED_TIP);
            super::super::probe("unoccupied", &r);
            // one request when the slider is let go (or clicked), not one per step of a drag
            if m != st.unoccupied && (r.drag_stopped() || !r.dragged()) {
                acts.push(Action::Unoccupied(m));
            }
            let status = payload.and_then(|o| o.unoccupied.as_ref());
            // while a change is pending the slider shows the new value; the lines speak of the request
            for l in unoccupied_lines(st.orbital_request().unoccupied, status, waiting) {
                ui.add(egui::Label::new(RichText::new(l).size(11.5).color(theme::INK_DIM)).wrap());
            }
        });
    });
}

/// The quick presets (user request 2026-10-20): one button each, the one the state matches
/// highlighted; "p only" needs rows of p character (a lone atom's), so it is off for a molecule.
fn presets(ui: &mut Ui, st: &UiState, payload: Option<&OrbitalDisplay>, acts: &mut Vec<Action>) {
    let rows = payload.map_or(&[][..], |o| &o.rows[..]);
    let active = super::super::actions::preset_active(st, rows);
    let has_p = rows.iter().any(|r| r.occupation > 0.0 && r.l == Some(1));
    let has_up = rows.iter().any(|r| r.occupation > 0.0 && r.spin == Spin::Up);
    ui.horizontal_wrapped(|ui| {
        ui.label("quick");
        for p in super::super::ColourPreset::ALL {
            use super::super::ColourPreset as P;
            let enabled = match p {
                P::POnly => has_p,
                P::OneSpin => has_up,
                P::Valence | P::Everything => true,
            };
            let r = ui.add_enabled(enabled, egui::Button::selectable(active == Some(p), p.title()));
            let r = if enabled { r.on_hover_text(preset_tip(p)) } else { r.on_disabled_hover_text(if p == P::POnly && !rows.is_empty() { P_ONLY_OFF } else { "needs a DFT step" }) };
            super::super::probe(&format!("preset-{}", p.word()), &r);
            if r.clicked() {
                acts.push(Action::ColourPreset(p));
            }
        }
    });
}

/// "p only" without p rows.
pub const P_ONLY_OFF: &str = "p only needs orbitals of p character about one nucleus: a lone atom's. A molecule's orbitals mix angular momenta about each nucleus, so none is a p orbital — tick the rows by hand";

/// What a preset includes.
pub fn preset_tip(p: super::super::ColourPreset) -> String {
    use super::super::ColourPreset as P;
    match p {
        P::Valence => format!("every occupied orbital but the core ones (more than {} Ha below the highest occupied: Li's 1s; a pseudised atom has none)", sim::CORE_GAP),
        P::OneSpin => "the occupied ↑ orbitals only: a closed shell's ↓ orbitals are the same shapes".into(),
        P::POnly => "the occupied orbitals of p character (both spins: a pair shares one colour)".into(),
        P::Everything => "every occupied orbital of both spins, core included (the default). Unoccupied orbitals are drawn only when ticked".into(),
    }
}

/// The list: a heading per channel, a line per orbital and its tags, then the notes. With usable
/// shares ([`grouped`]) the rows go under a heading per [`Group`], each group keeping the channel
/// structure (one ↑↓ list for a closed shell, else ↑ then ↓) and the occupied / unoccupied split.
fn list(ui: &mut Ui, st: &mut UiState, payload: Option<&OrbitalDisplay>, li: &ListInput<'_>, acts: &mut Vec<Action>) {
    super::group(ui, st, "d-orb-list", |ui, st| {
        ui.horizontal_wrapped(|ui| {
            theme::section(ui, "Orbitals");
            theme::info(ui, format!("label · spin · ε (Ha) · ε (eV) · occupation · largest atom share (in a molecule) · tags. {KS_NOTE} {HOMO_TIP}"));
        });
        let Some(o) = payload else {
            theme::note(ui, "needs a DFT step");
            return;
        };
        let levels = Levels::of(&o.rows).with_lone_atom(o.lone_atom).with_unoccupied(o.unoccupied.as_ref());
        if let Some(l) = levels_line(&levels) {
            ui.add(egui::Label::new(RichText::new(l).monospace().size(11.0)).wrap()).on_hover_text(format!("{HOMO_TIP}. {GAP_TIP}"));
        }
        // the warnings that qualify the list sit with it, above the rows (not below the fold)
        if let Some(l) = lone_atom_line(o.lone_atom) {
            let r = ui.add(egui::Label::new(RichText::new(l).size(11.5).color(super::super::AMBER)).wrap());
            if levels.inverted {
                r.on_hover_text(HOMO_INVERTED_TIP);
            }
        }
        if let Some(l) = smeared_molecule_line(o.lone_atom, li.z_sigma) {
            ui.add(egui::Label::new(RichText::new(l).size(11.5).color(super::super::AMBER)).wrap()).on_hover_text(SMEARED_MOLECULE_TIP);
        }
        if o.rows.is_empty() {
            theme::note(ui, "no electrons — nothing to show");
        }
        let by_atom = grouped(o, li.scene_z);
        let picked = match &o.what {
            OrbitalRequest::Orbital { spin, track, .. } => Some((*spin, *track)),
            _ => None,
        };
        let unocc_current = o.unoccupied.as_ref().is_none_or(|u| u.current);
        let [up, dn] = channels(&o.rows);
        if paired(&o.rows, &o.spin) {
            // a closed shell: one row per ↑↓ pair instead of two identical blocks (user request
            // 2026-11-11); any difference between the spins splits the list again
            let n_occ = up.iter().filter(|r| r.occupation > 0.0).count();
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("↑↓ paired · {n_occ} occupied")).strong());
                theme::info(ui, PAIRED_TIP);
            });
            let pairs = merged(&up, &dn);
            // by atom (user request 2026-11-12), a pair placed by its ↑ member's shares
            let ups: Vec<&OrbitalInfo> = pairs.iter().map(|(u, _)| *u).collect();
            let blocks: Vec<(Option<Owner>, usize, Vec<usize>)> = if by_atom {
                groups(&ups, li.scene_z, unocc_current).into_iter().map(|g| (Some(g.owner), g.others, g.rows)).collect()
            } else {
                vec![(None, 0, (0..pairs.len()).collect())]
            };
            for (owner, others, members) in blocks {
                if let Some(owner) = &owner {
                    group_heading(ui, owner, others, li.scene_z);
                }
                let mut shown_unocc = false;
                for i in members {
                    let (u, d) = pairs[i];
                    if u.occupation == 0.0 && !shown_unocc {
                        shown_unocc = true;
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("↑↓ unoccupied").color(theme::INK_DIM));
                            theme::info(ui, UNOCCUPIED_TIP);
                        });
                    }
                    // either member picked shows the pair as selected
                    let on = picked == Some((u.spin, u.track)) || d.is_some_and(|d| picked == Some((d.spin, d.track)));
                    let place = owner.as_ref().map_or_else(Place::default, |w| Place::of(w, u.shares.as_deref(), li.scene_z, li.selected));
                    row(ui, st, u, d, &levels, on, &place, acts);
                }
            }
        } else {
            let channel_line = |ui: &mut Ui, spin: Spin, rows: &[&OrbitalInfo]| {
                let n_occ = rows.iter().filter(|r| r.occupation > 0.0).count();
                if n_occ == 0 {
                    ui.label(RichText::new(format!("{}: none occupied", arrow(spin))).color(theme::INK_DIM));
                } else {
                    ui.label(RichText::new(format!("{} spin · {n_occ} occupied", arrow(spin))).strong());
                }
            };
            // one block per channel as before; by atom (user request 2026-11-12), the channels'
            // counts first, then each group with its ↑ rows then its ↓ rows
            let all: Vec<&OrbitalInfo> = up.iter().chain(dn.iter()).copied().collect();
            let blocks: Vec<(Option<Owner>, usize, Vec<usize>)> = if by_atom {
                channel_line(ui, Spin::Up, &up);
                channel_line(ui, Spin::Down, &dn);
                groups(&all, li.scene_z, unocc_current).into_iter().map(|g| (Some(g.owner), g.others, g.rows)).collect()
            } else {
                vec![(None, 0, (0..all.len()).collect())]
            };
            for (owner, others, members) in blocks {
                if let Some(owner) = &owner {
                    group_heading(ui, owner, others, li.scene_z);
                }
                for spin in [Spin::Up, Spin::Down] {
                    let rows: Vec<&OrbitalInfo> = members.iter().map(|&i| all[i]).filter(|r| r.spin == spin).collect();
                    match &owner {
                        None => channel_line(ui, spin, &rows),
                        Some(_) if rows.is_empty() => continue,
                        Some(_) => {
                            ui.label(RichText::new(format!("{} spin", arrow(spin))).size(11.0).color(theme::INK_DIM));
                        }
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
                        let place = owner.as_ref().map_or_else(Place::default, |w| Place::of(w, r.shares.as_deref(), li.scene_z, li.selected));
                        row(ui, st, r, None, &levels, picked == Some((r.spin, r.track)), &place, acts);
                    }
                }
            }
        }
        // why a molecule's list is not by atom, under the rows: the rows do not move when the
        // shares arrive and the note goes
        if let Some(n) = shares_note(o, li.scene_z, li.refs_pending) {
            theme::note(ui, n);
        }
        if o.rows.iter().any(|r| r.occupation == 0.0) {
            ui.add(egui::Label::new(RichText::new(GAP_TIP).size(11.0).color(theme::INK_FAINT)).wrap());
        }
    });
}

/// A group's heading in the grouped list, its tooltip saying what puts a row under it.
fn group_heading(ui: &mut Ui, owner: &Owner, others: usize, z: &[u8]) {
    ui.add_space(4.0);
    ui.label(RichText::new(heading(owner, others, z)).strong().color(theme::INK)).on_hover_text(heading_tip(owner));
}

/// One orbital: swatch, include box (all-coloured mode), the label as the picker, ε, occupation,
/// its largest atom share (`place`, in a molecule with shares), then its tags. With `partner` (the
/// ↓ member of a closed shell's pair, [`paired`]; user request 2026-11-11) `r` is the ↑ member and
/// the row stands for both: labelled `↑↓`, probed `orbital-↑↓-{↑ track}`, the occupations summed;
/// a click picks the ↑ orbital and the include box toggles both tracks. A row the selection does
/// not light (`place.faint`; user request 2026-11-12) is drawn in faint ink and stays clickable;
/// its warning tags keep their amber (a caveat is never hidden), and a picked row keeps its full
/// ink (faint text on the selection fill would be hard to read; Task 4 review, Minor 5).
#[allow(clippy::too_many_arguments)]
fn row(ui: &mut Ui, st: &mut UiState, r: &OrbitalInfo, partner: Option<&OrbitalInfo>, levels: &Levels, picked: bool, place: &Place, acts: &mut Vec<Action>) {
    let v = match partner {
        Some(d) => RowView::pair(r, d, levels),
        None => RowView::of(r, levels),
    };
    let arrows = if partner.is_some() { "↑↓" } else { arrow(r.spin) };
    let colour = providers::orbitals::ORBITAL_PALETTE[r.slot as usize % providers::orbitals::ORBITAL_PALETTE.len()];
    let faint = place.faint && !picked;
    // the secondary ink: dim, or faint with the rest of a row the selection does not light
    let dim = if faint { theme::INK_FAINT } else { theme::INK_DIM };
    ui.horizontal(|ui| {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
        let sw = swatch(colour);
        ui.painter().rect_filled(rect, 3.0, if faint { sw.gamma_multiply(0.35) } else { sw });
        resp.on_hover_text(if st.show == ShowMode::Coloured { "this orbital's colour in All orbitals coloured (a ↓ orbital shares its ↑ partner's colour)" } else { "this orbital's colour in All orbitals coloured" });
        if st.show == ShowMode::Coloured {
            // a pair's box is on while either member is drawn, and switches both
            let mut on = included(st, r) || partner.is_some_and(|d| included(st, d));
            let tip = match (partner.is_some(), r.occupation > 0.0) {
                (_, false) => UNOCC_INCLUDE_TIP,
                (true, true) => "include this pair (its ↑ and its ↓ orbital) in the colour volume",
                (false, true) => "include this orbital in the colour volume",
            };
            let cb = ui.checkbox(&mut on, "").on_hover_text(tip);
            super::super::probe(&format!("include-{}", r.track), &cb);
            if cb.changed() {
                acts.push(Action::Include { track: r.track, on });
                if let Some(d) = partner.filter(|d| d.track != r.track) {
                    acts.push(Action::Include { track: d.track, on });
                }
            }
        }
        let name = RichText::new(format!("{} {arrows}", v.label)).monospace();
        let name = if faint { name.color(theme::INK_FAINT) } else { name };
        let b = ui.add(egui::Button::selectable(picked, name));
        let b = b.on_hover_text(v.label_tip());
        super::super::probe(&format!("orbital-{arrows}-{}", r.track), &b);
        if b.clicked() {
            acts.push(Action::PickOrbital { spin: r.spin, track: r.track });
        }
        let ha = RichText::new(&v.ha).monospace();
        ui.label(if faint { ha.color(theme::INK_FAINT) } else { ha }).on_hover_text("ε in Hartree: a Kohn–Sham eigenvalue; only the HOMO's has a physical meaning, and in LSDA it is too shallow");
        ui.label(RichText::new(&v.ev).monospace().color(dim)).on_hover_text("ε in eV (1 Ha = 27.2114 eV; converted for display only)");
        let occ_tip = if partner.is_some() {
            "occupation of the pair: the sum of its ↑ and its ↓ orbital's (1 + 1 = 2 for a closed shell; a fraction is a degenerate shell shared equally, summed over both spins — pseudopotentials: the spherical-atom ensemble, not a measurable orbital occupation)"
        } else {
            "occupation: 1 or 0 (integer aufbau); a fraction (1/3, …) is a degenerate shell shared equally (pseudopotentials: the spherical-atom ensemble, not a measurable orbital occupation)"
        };
        ui.label(RichText::new(format!("occ {}", v.occupation)).monospace().color(dim)).on_hover_text(occ_tip);
        if let Some((text, hover)) = &place.share {
            // set off from the occupation ("occ 2 · O #0 79 %": the controller's screenshot read
            // "occ 2 O #0 79 %" as one run)
            ui.label(RichText::new("·").monospace().color(dim));
            let s = ui.label(RichText::new(text).monospace().color(dim)).on_hover_text(hover);
            super::super::probe(&format!("share-{arrows}-{}", r.track), &s);
        }
    });
    if !v.tags.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.add_space(18.0);
            for t in &v.tags {
                let c = if t.warn { super::super::AMBER } else { dim };
                ui.label(RichText::new(&t.text).size(11.0).color(c)).on_hover_text(&t.tip);
            }
        });
    }
}
