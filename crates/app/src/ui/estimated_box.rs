//! The estimated solver's bigger, physics-sized box (user decision 2026-11-14; the rule and its
//! derivation: `crate::quality`, "The estimated solver's box"). When the app's own scene is
//! (re)built under the Estimated backend — the switch to it, a preset loaded in it, a Quality
//! change in it — the grid grows to L_est = extent + 2 (R_cloud + ROOM), never below the preset's
//! box, so the molecule can move and its teacher samples stay clear of the box rule. The DFT
//! backends keep today's grids (leaving the estimated solver returns a grid it grew to the base
//! box), and a scene file, a built scene or a segment keeps its own grid: [`note`] then says when
//! the box is short and offers the regrid, through `Action::Quality` (so a running solve is
//! cancelled as for any quality change and `--record` sees it). Atoms placed or dragged later
//! never regrid by themselves: the same note.
use super::{default_wall_half_width, Action, PanelInput, PanelOutput, UiState, BOX_MARGIN_WARN};
use crate::quality::{box_len, estimated_box, gpu_solvers, regrid, snap, vram_budget, vram_check, EstimatedBox};
use crate::scene::Scene;
use crate::sim::{Backend, Command};
use sim::DynCommand;

/// The snapshot's isolated-atom profiles (`Snapshot::atom_profiles`, the estimated solver's own
/// solves) into `st.cloud_profiles`: the latest per (Z, ionic policy), kept across grids so the
/// next scene's R_cloud comes from the solver's answer, not the provisional bound.
pub fn refresh_profiles(st: &mut UiState, input: &PanelInput<'_>) {
    let Some(s) = input.snapshot.filter(|s| s.backend == Backend::Estimated) else { return };
    let from = std::sync::Arc::as_ptr(&s.atom_profiles) as usize;
    if from == st.cloud_profiles_from {
        return;
    }
    st.cloud_profiles_from = from;
    for p in s.atom_profiles.iter() {
        st.cloud_profiles.retain(|q| (q.z, q.ionic) != (p.z, p.ionic));
        st.cloud_profiles.push(p.clone());
    }
}

/// The GPU memory guard's verdict on `s` at n³ for the estimated solver (teacher + auditor).
fn fits(st: &UiState, s: &Scene) -> impl Fn(usize) -> bool {
    let (solvers, budget, s) = (gpu_solvers(Backend::Estimated, st.gpu_available), vram_budget(), s.clone());
    move |n| vram_check(&Scene { grid_n: n, ..s.clone() }, solvers, budget).is_ok()
}

/// `s` on the estimated box at spacing h (snapped) over the base box `base_len`.
pub fn grow(st: &UiState, s: &Scene, base_len: f64, h: f64) -> (Scene, EstimatedBox) {
    let h = snap(h);
    let eb = estimated_box(s, base_len, h, &st.cloud_profiles, fits(st, s));
    (Scene { grid_n: eb.n, grid_h: h, ..s.clone() }, eb)
}

/// The container follows a cell that changed from `old` to `new` when it sat at the old cell's
/// default (1.5 Bohr inside the face: the walls' rule is unchanged); a width the user chose stays.
/// `send`: also send it to the running worker (a respawned one gets it with the run settings).
fn walls_follow(st: &mut UiState, old: &Scene, new: &Scene, send: bool, out: &mut PanelOutput) {
    if (st.wall_half_width - default_wall_half_width(old)).abs() < 1e-9 {
        st.wall_half_width = default_wall_half_width(new);
        if send && st.walls_on {
            out.commands.push(Command::Dyn(DynCommand::SetWalls(st.walls())));
        }
    }
}

/// "the box grows to 16.2 Bohr (109³ × 0.15)" and, when the memory guard cut it, why.
fn grown_words(eb: &EstimatedBox, h: f64) -> String {
    let mut t = format!("the box grows to {:.1} Bohr ({}³ × {h}) so the molecule can move and its samples stay clear of the walls", eb.len(h), eb.n);
    if eb.capped {
        t += &format!(" — the rule asks ≈ {:.1} Bohr ({}³); the GPU memory guard allows {}³", eb.want_len, crate::quality::grid_n(eb.want_len, h), eb.n);
    }
    t
}

/// `Action::Backend` changed the solver from `was` to `st.backend`. To Estimated: a preset's grid
/// grows to the estimated box (the new worker starts on it: `PanelOutput::switch_scene`). Away from
/// it: a grid this rule grew returns to the base box, unless a nucleus would then sit within the
/// warning margin of the smaller box's face (it stays, and the status line says so).
pub(super) fn on_switch(st: &mut UiState, input: &PanelInput<'_>, was: Backend, out: &mut PanelOutput) {
    let scene = input.scene;
    let h = scene.grid_h;
    if st.backend == Backend::Estimated {
        if st.source.preset().is_none() {
            return;
        }
        let (s, eb) = grow(st, scene, st.box_len, h);
        if s.grid_n == scene.grid_n || super::actions::vram_check(st, &s).is_err() {
            return;
        }
        (st.grid_n, st.grid_h) = (s.grid_n, s.grid_h);
        st.estimated_grid = Some((s.grid_n, s.grid_h));
        walls_follow(st, scene, &s, false, out);
        st.set_status(format!("solver switched to {}: the run restarts paused at t = 0; {}", Backend::Estimated.label(), grown_words(&eb, h)));
        out.switch_scene = Some(s);
    } else if was == Backend::Estimated && st.estimated_grid == Some((scene.grid_n, h)) {
        st.estimated_grid = None;
        let s = regrid(scene, st.box_len, h);
        if s.grid_n == scene.grid_n {
            return;
        }
        let half = box_len(&s) / 2.0;
        if scene.nuclei.iter().any(|n| n.pos.iter().any(|x| x.abs() > half - BOX_MARGIN_WARN)) {
            st.set_status(format!("the estimated solver's {:.1} Bohr box is kept: a nucleus is outside the {:.1} Bohr box the presets use (Scene ▸ Quality returns to it)", box_len(scene), box_len(&s)));
            return;
        }
        (st.grid_n, st.grid_h) = (s.grid_n, s.grid_h);
        walls_follow(st, scene, &s, false, out);
        st.set_status(format!("solver switched to {}: the run restarts paused at t = 0 on the {:.1} Bohr box ({}³)", st.backend.label(), box_len(&s), s.grid_n));
        out.switch_scene = Some(s);
    }
}

/// `Action::Preset` under the estimated solver: the preset at quality h grown to the estimated box
/// over its own box `file_box`; `estimated_grid` records it.
pub(super) fn preset(st: &mut UiState, s: Scene, file_box: f64) -> Scene {
    if st.backend != Backend::Estimated {
        st.estimated_grid = None;
        return s;
    }
    let (g, _) = grow(st, &s, file_box, s.grid_h);
    st.estimated_grid = (g.grid_n != s.grid_n).then_some((g.grid_n, g.grid_h));
    g
}

/// `Action::Quality` at h: the base regrid (the box `box_len` kept), grown to the estimated box
/// under the estimated solver; and whether the container should follow it (a grid this rule grows
/// or returns: [`walls_follow`], sent by the caller after the grid).
pub(super) fn quality(st: &UiState, input: &PanelInput<'_>, h: f64) -> (Scene, bool) {
    let base = regrid(input.scene, st.box_len, h);
    let s = if st.backend == Backend::Estimated { grow(st, &base, st.box_len, h).0 } else { base.clone() };
    let was_grown = st.estimated_grid == Some((input.scene.grid_n, input.scene.grid_h));
    let follow = s.grid_n != base.grid_n || was_grown;
    (s, follow)
}

/// The container after an accepted grid change from `old` to `new` ([`walls_follow`], sent).
pub(super) fn walls_after_grid(st: &mut UiState, old: &Scene, new: &Scene, out: &mut PanelOutput) {
    walls_follow(st, old, new, true, out);
}

/// After an accepted `Action::Quality`: the grid it set is the rule's when it differs from the
/// base box's.
pub(super) fn quality_applied(st: &mut UiState, s: &Scene) {
    let base_n = crate::quality::grid_n(st.box_len, s.grid_h);
    st.estimated_grid = (st.backend == Backend::Estimated && s.grid_n != base_n).then_some((s.grid_n, s.grid_h));
}

/// The box Scene ▸ Quality would build at spacing h (Bohr): the base box, or under the estimated
/// solver the estimated one (the menu's n per level).
pub fn quality_box(st: &UiState, scene: &Scene, h: f64) -> f64 {
    if st.backend != Backend::Estimated {
        return st.box_len;
    }
    let eb = estimated_box(scene, st.box_len, h, &st.cloud_profiles, |_| true);
    eb.want_len
}

/// The note under the estimated solver when the scene's box is shorter than the rule asks by more
/// than h (a scene file's own grid, atoms placed or dragged since, a profile that refined R_cloud):
/// the line, its hover, and the grid (n, h) the button regrids to (`None`: the GPU memory guard
/// allows nothing bigger at this spacing).
#[derive(Clone, Debug, PartialEq)]
pub struct BoxNote {
    pub text: String,
    pub tip: String,
    pub regrid: Option<(usize, f64)>,
}

/// [`BoxNote`] for `scene`, or `None` (another solver, no nuclei, the box is big enough). Cheap: the
/// extent from the positions and R_cloud from the cached profiles; the memory guard runs only when
/// the wanted grid changes (`estimated_note_cache`).
pub fn note(st: &mut UiState, scene: &Scene) -> Option<BoxNote> {
    if st.backend != Backend::Estimated || scene.nuclei.is_empty() {
        return None;
    }
    let h = scene.grid_h;
    let now = box_len(scene);
    let free = estimated_box(scene, now, h, &st.cloud_profiles, |_| true);
    if free.want_len <= now + h {
        return None;
    }
    let zs = scene.nuclei.iter().fold(0u64, |a, n| a.wrapping_mul(131).wrapping_add(n.z as u64));
    let key = (free.want_n, h, scene.grid_n, scene.ionic, zs);
    let n = match st.estimated_note_cache {
        Some((k, n)) if k == key => n,
        _ => {
            let n = estimated_box(scene, now, h, &st.cloud_profiles, fits(st, scene)).n;
            st.estimated_note_cache = Some((key, n));
            n
        }
    };
    let mut text = format!("box is {now:.1} Bohr; samples near the walls can't be learned — a bigger grid (≈ {:.1} Bohr) avoids it", free.want_len);
    if n < free.want_n {
        if n > scene.grid_n {
            text += &format!(" (the GPU memory guard allows {n}³ at h {h})");
        } else {
            text += " (the GPU memory guard allows no bigger grid at this spacing: a coarser quality, or fewer atoms)";
        }
    }
    let species = free.cloud.z.map_or_else(String::new, |z| format!(" of {}", units::isotopes::symbol(z)));
    let origin = if free.cloud.provisional {
        "a provisional bound — a hydrogen-like tail — until every species' isolated atom has been solved"
    } else {
        "read off the solver's own isolated-atom density"
    };
    let tip = format!(
        "a sample is box-affected (not learned) when the teacher's density on the box's outermost plane exceeds the box rule; the estimated box is the molecule's span {:.2} Bohr + 2 × (R_cloud {:.2} Bohr{species}, where the isolated atom's density falls below the rule — {origin} — + {:.1} Bohr of room). The grid stays until you regrid; a DFT solve at {n}³ costs ≈ {:.1}× this grid's.",
        free.extent,
        free.cloud.r,
        crate::quality::ESTIMATED_ROOM,
        crate::quality::scaled_cost(1.0, scene.grid_n, n)
    );
    Some(BoxNote { text, tip, regrid: (n > scene.grid_n).then_some((n, h)) })
}

/// The note's row (Readouts ▸ health): the amber line and its button, which regrids through
/// `Action::Quality` at the scene's spacing.
pub fn show_note(ui: &mut egui::Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let Some(n) = note(st, input.scene) else { return };
    ui.horizontal_wrapped(|ui| {
        super::theme::light(ui, super::AMBER);
        ui.label(egui::RichText::new(&n.text).size(11.5).color(super::AMBER)).on_hover_text(&n.tip);
        if let Some((g, h)) = n.regrid {
            let r = super::probed("estimated-box-regrid", ui.small_button(format!("Use {g}³ ({:.1} Bohr)", (g as f64 - 1.0) * h)));
            if r.on_hover_text("regrid at this spacing to the estimated box (Scene ▸ Quality: the scene is re-solved, the nuclei stay where they are)").clicked() {
                acts.push(Action::Quality(h));
            }
        }
    });
}
