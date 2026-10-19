//! Loading a model whose settings differ from the scene's (user request 2026-10-18: "When loading
//! models, it should give an option to load the required scene settings to run it, instead of
//! only refusing to load it").
//!
//! A picked `.model` file's provenance (`surrogate::sidecar::read_provenance`: the header only) is
//! compared with what the scene and the app would produce: the same fields the driver compares
//! (`surrogate::Provenance`'s `PartialEq`: grid n and h, σ, tolerance, charge, multiplicity,
//! teacher, ionic surface), plus the recorded isotopes, which are shown but never compared (M6
//! ruling P2). Equal → the load goes out as before. Different → [`offer`] builds a
//! [`ModelOffer`]: one [`Row`] per differing setting ("model: X · scene: Y") and the [`Edit`]s
//! that would make the scene match. The card in the 3D view (`overlays`) shows the rows; "Apply
//! the model's settings and load" ([`Action::ApplyModelSettings`](super::Action)) recomputes the
//! offer against the scene as it is then and sends the edits through the ordinary commands
//! (`SetGrid`, `SetSigma`, `SetCharge`, `SetMultiplicity`, `SetIonicPolicy`, `SetIsotope`) followed
//! by `LoadModel`, so `--record` and the worker see nothing new.
//!
//! What cannot be applied is named with its reason on its row and disables Apply (the card never
//! offers a partial match that the driver would still refuse):
//! - the **teacher tolerance**: fixed by the app (`worker::SAMPLE_TOL`), not a setting;
//! - the **teacher**: the estimated solver teaches with the GPU solver when the machine has a GPU,
//!   else the CPU one — it is a fact of the session, not a switch;
//! - an **ionic surface** this build cannot reproduce (another recipe hash or occupation scheme),
//!   or one the scene's species cannot reach under the policy it names;
//! - a **grid** (or policy) the GPU memory guard refuses, or an invalid grid;
//! - a **charge or multiplicity** the scene's electrons cannot realise.
//! The grid, σ, charge, multiplicity and policy are applied; the nuclei stay where they are.
//!
//! Pure functions over data (no egui, no worker) so the diff and the plan are unit tested.
use crate::scene::{validate_grid, Scene};
use std::path::{Path, PathBuf};
use surrogate::Provenance;

/// One setting that differs between the model and the scene.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// "grid", "σ", "charge", "multiplicity", "teacher", "teacher tolerance", "ionic potential",
    /// "isotopes".
    pub name: &'static str,
    /// What the model was taught at.
    pub model: String,
    /// What the scene (or the session) has.
    pub scene: String,
    /// Why this cannot be applied; `Some` disables Apply.
    pub blocked: Option<String>,
    /// An explanation that does not block (the isotope rows, the pseudopotential reach).
    pub note: Option<String>,
    /// `false`: recorded, not compared — the driver loads the model regardless (M6 ruling P2).
    pub compared: bool,
}

/// One change of the scene that Apply sends, in the order it is sent.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    Grid { n: usize, h: f64 },
    Sigma(Option<f64>),
    Charge(i32),
    Multiplicity(Option<u32>),
    Ionic(sim::IonicPolicy),
    Isotope { index: usize, a: u16 },
}

/// A model file whose settings differ from the scene's, and what it would take to run it.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOffer {
    pub path: PathBuf,
    pub rows: Vec<Row>,
    /// What Apply sends, in order; unused while a row is blocked.
    pub edits: Vec<Edit>,
}

impl ModelOffer {
    /// The first reason Apply is unavailable, if any.
    pub fn blocked(&self) -> Option<&str> {
        self.rows.iter().find_map(|r| r.blocked.as_deref())
    }
}

impl Row {
    /// "model: X · scene: Y", as the card prints it beside the setting's name.
    pub fn versus(&self) -> String {
        format!("model: {} · scene: {}", self.model, self.scene)
    }
}

/// What [`offer`] needs to know of the session besides the scene.
pub struct Context<'a> {
    /// The teacher's name as the provider reports it ("Kohn-Sham (GPU)").
    pub teacher: &'a str,
    /// The tolerance the estimated solver's teacher solves at (`worker::SAMPLE_TOL`).
    pub sample_tol: f64,
    /// Some species of the scene is pseudised under its policy (the snapshot's evaluation, else the
    /// cache — `ui::pseudo::pseudised`).
    pub pseudised: bool,
    /// The GPU memory guard (`quality::vram_check`) on the scene the edits would produce.
    pub vram: &'a dyn Fn(&Scene) -> Result<(), String>,
    /// What making a species' pseudopotential available would cost now (`providers::pseudo::generation_cost`): the ionic
    /// row says when Apply would start the dimer gate on the CPU (≈ 40 min; final review M6).
    pub generation_cost: &'a dyn Fn(u8) -> providers::pseudo::GenerationCost,
}

/// The note of an ionic row whose Apply would generate H or He through the dimer gate on the CPU (≈ 40 min each, none
/// cached here or on disk): which species, and that the solve waits for it.
fn cpu_gate_note(scene: &Scene, cost: &dyn Fn(u8) -> providers::pseudo::GenerationCost) -> Option<String> {
    let mut zs: Vec<u8> = scene.nuclei.iter().map(|n| n.z).filter(|&z| z <= qm::radial::ledger::LEDGER.dimer_z_max).collect();
    zs.sort_unstable();
    zs.dedup();
    let slow: Vec<&str> = zs.into_iter().filter(|&z| cost(z) == providers::pseudo::GenerationCost::CpuGate).map(units::isotopes::symbol).collect();
    if slow.is_empty() {
        return None;
    }
    Some(format!(
        "applying generates the {} pseudopotential{} on the CPU first — about {} min (the dimer check; this machine has no GPU for it), and the first solve waits for it",
        slow.join(" and "),
        if slow.len() > 1 { "s" } else { "" },
        40 * slow.len()
    ))
}

/// Teacher name for the card: "Kohn-Sham (GPU)" → "GPU solver".
fn teacher_label(t: &str) -> String {
    match t.strip_prefix("Kohn-Sham (").and_then(|r| r.strip_suffix(')')) {
        Some(k) => format!("{k} solver"),
        None => t.to_string(),
    }
}

fn grid_label(n: usize, h: f64) -> String {
    format!("{n}³ × h {h} Bohr")
}

fn mult_label(m: Option<u32>) -> String {
    m.map_or("lowest".to_string(), |m| m.to_string())
}

/// The ionic tag in words: "none" → all-electron; "cores/…" and "all/…" with their policy.
pub fn ionic_label(tag: &str) -> String {
    match tag.split('/').next().and_then(sim::IonicPolicy::parse) {
        Some(sim::IonicPolicy::None) => "all-electron (none)".to_string(),
        Some(sim::IonicPolicy::CoresOnly) => format!("pseudopotentials on species with a core ({})", short_tag(tag)),
        Some(sim::IonicPolicy::All) => format!("pseudopotentials on every species ({})", short_tag(tag)),
        // a tag carries the *surface* word (`All` or `CoresOnly` for an accepted-species scene), never "accepted"; defensive
        Some(sim::IonicPolicy::Accepted) => format!("pseudopotentials on every species the recipe accepts ({})", short_tag(tag)),
        None => format!("unknown ({tag})"),
    }
}

/// "cores/8a3f…" — the word and the head of the recipe hash.
fn short_tag(tag: &str) -> String {
    let mut parts = tag.split('/');
    match (parts.next(), parts.next()) {
        (Some(w), Some(h)) => format!("{w}/{}…", h.chars().take(4).collect::<String>()),
        (Some(w), None) => w.to_string(),
        _ => tag.to_string(),
    }
}

/// The scene's isotope set as the provenance records it: sorted unique (Z, A).
fn scene_masses(s: &Scene) -> Vec<(u8, u16)> {
    let mut m: Vec<(u8, u16)> = s.nuclei.iter().filter_map(|n| n.mass_number().map(|a| (n.z, a))).collect();
    m.sort_unstable();
    m.dedup();
    m
}

fn masses_label(m: &[(u8, u16)]) -> String {
    if m.is_empty() {
        "none".to_string()
    } else {
        m.iter().map(|&(z, a)| units::isotopes::isotope_label(z, a)).collect::<Vec<_>>().join(" ")
    }
}

/// Compares the model's provenance with the scene. `None` when nothing the driver compares
/// differs (the load goes out as it always did, whatever the recorded isotopes say).
pub fn offer(path: &Path, prov: &Provenance, scene: &Scene, ctx: &Context<'_>) -> Option<ModelOffer> {
    let mut rows: Vec<Row> = Vec::new();
    let mut edits: Vec<Edit> = Vec::new();
    let row = |name, model: String, scene: String, compared| Row { name, model, scene, blocked: None, note: None, compared };
    // the scene as the edits would leave it, for the later checks
    let mut t = scene.clone();

    // grid
    let grid_differs = (prov.grid_n, prov.h) != (scene.grid_n, scene.grid_h);
    let mut grid_row = None;
    if grid_differs {
        let mut r = row("grid", grid_label(prov.grid_n, prov.h), grid_label(scene.grid_n, scene.grid_h), true);
        r.blocked = validate_grid(prov.grid_n, prov.h).err().map(|e| format!("the model's grid is not valid here: {e}"));
        (t.grid_n, t.grid_h) = (prov.grid_n, prov.h);
        edits.push(Edit::Grid { n: prov.grid_n, h: prov.h });
        grid_row = Some(rows.len());
        rows.push(r);
    }

    // σ: judged after the grid change (an automatic σ follows h)
    // (a row only when an edit is needed: σ = h on both sides at different h differ in value but not in setting)
    let sigma_now = scene.sigma.unwrap_or(scene.grid_h);
    if t.sigma.unwrap_or(t.grid_h) != prov.sigma {
        // σ = h is the automatic setting
        let new = if prov.sigma == prov.h { None } else { Some(prov.sigma) };
        t.sigma = new;
        edits.push(Edit::Sigma(new));
        rows.push(row("σ", format!("{} Bohr", prov.sigma), format!("{sigma_now} Bohr"), true));
    }

    // charge and multiplicity (can the scene's electrons realise them?)
    if prov.charge != scene.charge {
        let mut r = row("charge", prov.charge.to_string(), scene.charge.to_string(), true);
        t.charge = prov.charge;
        if !t.nuclei.is_empty() && t.nuclei.iter().map(|n| n.z as i64).sum::<i64>() - (t.charge as i64) < 0 {
            r.blocked = Some(format!("a charge of {} leaves this scene's nuclei with negative electrons", prov.charge));
        }
        edits.push(Edit::Charge(prov.charge));
        rows.push(r);
    }
    if prov.multiplicity != scene.multiplicity {
        let mut r = row("multiplicity", mult_label(prov.multiplicity), mult_label(scene.multiplicity), true);
        t.multiplicity = prov.multiplicity;
        let n = t.nuclei.iter().map(|n| n.z as i64).sum::<i64>() - t.charge as i64;
        if let (Some(m), false) = (prov.multiplicity, t.nuclei.is_empty()) {
            let m = m as i64;
            if m < 1 || m > n + 1 || (n + 1 - m) % 2 != 0 {
                r.blocked = Some(format!("a multiplicity of {m} is impossible for {n} electrons"));
            }
        }
        edits.push(Edit::Multiplicity(prov.multiplicity));
        rows.push(r);
    }

    // teacher and its tolerance: facts of the session
    if prov.teacher != ctx.teacher {
        let mut r = row("teacher", teacher_label(&prov.teacher), teacher_label(ctx.teacher), true);
        r.blocked = Some(format!(
            "the estimated solver teaches with the {} on this machine (the GPU solver when there is a GPU, else the CPU one) — it is not a setting",
            teacher_label(ctx.teacher)
        ));
        rows.push(r);
    }
    if prov.tol != ctx.sample_tol {
        let mut r = row("teacher tolerance", format!("{:e}", prov.tol), format!("{:e}", ctx.sample_tol), true);
        r.blocked = Some(format!("the teacher's SCF tolerance is fixed by the app at {:e} (labels are permanent); it is not a setting", ctx.sample_tol));
        rows.push(r);
    }

    // the ionic surface
    let scene_tag = if ctx.pseudised { providers::pseudo::tag_of(scene.ionic) } else { "none".to_string() };
    let mut ionic_row = None;
    if prov.ionic != scene_tag {
        let mut r = row("ionic potential", ionic_label(&prov.ionic), ionic_label(&scene_tag), true);
        match prov.ionic.split('/').next().and_then(sim::IonicPolicy::parse) {
            None => r.blocked = Some(format!("the ionic tag '{}' is not one this build knows", prov.ionic)),
            Some(sim::IonicPolicy::None) => {
                t.ionic = sim::IonicPolicy::None;
                edits.push(Edit::Ionic(sim::IonicPolicy::None));
            }
            Some(p) if prov.ionic != providers::pseudo::tag_of(p) => {
                r.blocked = Some("trained with a different pseudopotential recipe than this build's (another recipe version or occupation scheme): its surface cannot be reproduced here — sample this scene instead".to_string());
            }
            Some(p) if p == scene.ionic => {
                r.blocked = Some(format!("no species in this scene is pseudised under '{}' (nothing here has a core), so no setting reaches that surface", p.word()));
            }
            Some(p) => {
                t.ionic = p;
                edits.push(Edit::Ionic(p));
                if p == sim::IonicPolicy::CoresOnly {
                    r.note = Some("takes effect only for species that have a core (not H, He)".to_string());
                }
                // `All` pseudises H and He whatever the backend: without a GPU their dimer gate runs on the CPU (final
                // review M6 — said on the row, never only after the click; `Accepted` keeps them all-electron there)
                if p == sim::IonicPolicy::All {
                    r.note = cpu_gate_note(scene, ctx.generation_cost);
                }
            }
        }
        ionic_row = Some(rows.len());
        rows.push(r);
    }

    // nothing the driver compares differs: the load needs no card
    if rows.is_empty() {
        return None;
    }

    // the GPU memory guard on the scene the edits produce
    if let Err(e) = (ctx.vram)(&t) {
        if let Some(i) = grid_row.or(ionic_row) {
            rows[i].blocked.get_or_insert(e);
        }
    }

    // the recorded isotopes: shown beside the others, never compared, never blocking
    let model_masses = &prov.masses;
    let have = scene_masses(scene);
    if !model_masses.is_empty() && *model_masses != have {
        let mut r = row("isotopes", masses_label(model_masses), masses_label(&have), false);
        let mut iso = Vec::new();
        let mut mixed = false;
        for z in model_masses.iter().map(|m| m.0).collect::<std::collections::BTreeSet<_>>() {
            let ours: Vec<u16> = model_masses.iter().filter(|m| m.0 == z).map(|m| m.1).collect();
            if ours.len() != 1 {
                if scene.nuclei.iter().any(|n| n.z == z) {
                    mixed = true;
                }
                continue;
            }
            for (index, n) in scene.nuclei.iter().enumerate().filter(|(_, n)| n.z == z) {
                if n.mass_number() != Some(ours[0]) && units::isotope_mass_amu_a(z, ours[0]).is_some() {
                    iso.push(Edit::Isotope { index, a: ours[0] });
                }
            }
        }
        r.note = Some(match (iso.is_empty(), mixed) {
            (true, true) => "recorded, not compared: the model's samples mixed isotopes of an element, so none is assigned to the atoms".to_string(),
            (true, false) => "recorded, not compared: the model's elements are not in this scene, or have no such isotope".to_string(),
            (false, true) => "recorded, not compared: single-isotope elements are set to the model's; an element it mixed is left as it is".to_string(),
            (false, false) => "recorded, not compared: the scene's atoms take the model's isotope (the Born-Oppenheimer surface does not depend on it)".to_string(),
        });
        edits.extend(iso);
        rows.push(r);
    }

    Some(ModelOffer { path: path.to_path_buf(), rows, edits })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneNucleus;

    fn nucleus(z: u8, a: Option<u16>, x: f64) -> SceneNucleus {
        SceneNucleus { z, a, pos: [x, 0.0, 0.0], vel: [0.0; 3] }
    }

    fn scene() -> Scene {
        Scene {
            nuclei: vec![nucleus(1, None, -0.725), nucleus(1, None, 0.725)],
            charge: 0,
            multiplicity: None,
            grid_n: 81,
            grid_h: 0.15,
            sigma: None,
            ionic: sim::IonicPolicy::CoresOnly,
        }
    }

    fn prov_of(s: &Scene) -> Provenance {
        Provenance {
            grid_n: s.grid_n,
            h: s.grid_h,
            sigma: s.sigma.unwrap_or(s.grid_h),
            tol: 1e-6,
            charge: s.charge,
            multiplicity: s.multiplicity,
            teacher: "Kohn-Sham (GPU)".into(),
            masses: vec![(1, 1)],
            ionic: "none".into(),
        }
    }

    fn ok(_: &Scene) -> Result<(), String> {
        Ok(())
    }

    fn ctx<'a>(vram: &'a dyn Fn(&Scene) -> Result<(), String>) -> Context<'a> {
        Context { teacher: "Kohn-Sham (GPU)", sample_tol: 1e-6, pseudised: false, vram, generation_cost: &|_| providers::pseudo::GenerationCost::Seconds }
    }

    fn make(prov: &Provenance, s: &Scene) -> Option<ModelOffer> {
        offer(Path::new("x.model"), prov, s, &ctx(&ok))
    }

    #[test]
    fn matching_settings_need_no_card_and_recorded_isotopes_alone_never_make_one() {
        let s = scene();
        assert_eq!(make(&prov_of(&s), &s), None);
        // D2 model on an H2 scene: the masses are recorded, not compared (ruling P2)
        let mut p = prov_of(&s);
        p.masses = vec![(1, 2)];
        assert_eq!(make(&p, &s), None);
        // an old file with no mass record
        p.masses.clear();
        assert_eq!(make(&p, &s), None);
    }

    #[test]
    fn every_differing_setting_gets_a_row_with_both_values() {
        let s = scene();
        let mut p = prov_of(&s);
        p.grid_n = 49;
        p.h = 0.25;
        p.sigma = 0.3;
        p.charge = 1;
        p.multiplicity = Some(2);
        let o = make(&p, &s).expect("differs");
        let lines: Vec<String> = o.rows.iter().map(|r| format!("{} — {}", r.name, r.versus())).collect();
        assert_eq!(
            lines,
            [
                "grid — model: 49³ × h 0.25 Bohr · scene: 81³ × h 0.15 Bohr",
                "σ — model: 0.3 Bohr · scene: 0.15 Bohr",
                "charge — model: 1 · scene: 0",
                "multiplicity — model: 2 · scene: lowest",
            ]
        );
        assert_eq!(o.blocked(), None);
        assert_eq!(
            o.edits,
            vec![Edit::Grid { n: 49, h: 0.25 }, Edit::Sigma(Some(0.3)), Edit::Charge(1), Edit::Multiplicity(Some(2))]
        );
    }

    #[test]
    fn an_automatic_sigma_follows_the_grid_and_an_explicit_one_is_reset() {
        // σ = h in the model, the scene has an explicit σ = 0.3: back to automatic
        let mut s = scene();
        s.sigma = Some(0.3);
        let mut p = prov_of(&scene());
        p.grid_n = 49;
        p.h = 0.25;
        p.sigma = 0.25;
        let o = make(&p, &s).unwrap();
        assert_eq!(o.edits, vec![Edit::Grid { n: 49, h: 0.25 }, Edit::Sigma(None)]);
        // σ = h both sides, only the grid differs: σ follows h by itself, no σ edit and no σ row
        let s = scene();
        let o = make(&p, &s).unwrap();
        assert_eq!(o.edits, vec![Edit::Grid { n: 49, h: 0.25 }]);
        assert!(o.rows.iter().all(|r| r.name != "σ"));
        // the scene's σ equals the model's σ today but the grid change would move an automatic one
        let mut p = prov_of(&s);
        p.grid_n = 49;
        p.h = 0.25;
        p.sigma = 0.15;
        let o = make(&p, &s).unwrap();
        assert_eq!(o.edits, vec![Edit::Grid { n: 49, h: 0.25 }, Edit::Sigma(Some(0.15))]);
        assert!(o.rows.iter().any(|r| r.name == "σ"), "{:?}", o.rows);
    }

    #[test]
    fn the_tolerance_and_the_teacher_are_named_and_block_apply() {
        let s = scene();
        let mut p = prov_of(&s);
        p.tol = 1e-5;
        p.teacher = "Kohn-Sham (CPU)".into();
        let o = make(&p, &s).unwrap();
        let tol = o.rows.iter().find(|r| r.name == "teacher tolerance").unwrap();
        assert_eq!((tol.model.as_str(), tol.scene.as_str()), ("1e-5", "1e-6"));
        assert!(tol.blocked.as_deref().unwrap().contains("fixed by the app"));
        let t = o.rows.iter().find(|r| r.name == "teacher").unwrap();
        assert_eq!((t.model.as_str(), t.scene.as_str()), ("CPU solver", "GPU solver"));
        assert!(t.blocked.is_some());
        assert!(o.blocked().is_some());
        assert!(o.edits.is_empty(), "nothing the scene could change");
    }

    #[test]
    fn the_ionic_tag_maps_to_a_policy_or_names_why_not() {
        let mut s = scene();
        s.ionic = sim::IonicPolicy::All;
        let c = Context { pseudised: true, ..ctx(&ok) };
        let all_tag = providers::pseudo::tag_of(sim::IonicPolicy::All);
        // model all-electron, scene pseudised: policy none
        let o = offer(Path::new("x.model"), &prov_of(&s), &s, &c).unwrap();
        assert_eq!(o.edits, vec![Edit::Ionic(sim::IonicPolicy::None)]);
        assert_eq!(o.rows[0].model, "all-electron (none)");
        assert!(o.rows[0].scene.starts_with("pseudopotentials on every species (all/"), "{}", o.rows[0].scene);
        // model pseudised under 'cores', scene 'all': set the policy; the note says it needs a core
        let mut p = prov_of(&s);
        p.ionic = providers::pseudo::tag_of(sim::IonicPolicy::CoresOnly);
        let o = offer(Path::new("x.model"), &p, &s, &c).unwrap();
        assert_eq!(o.edits, vec![Edit::Ionic(sim::IonicPolicy::CoresOnly)]);
        assert!(o.rows[0].model.starts_with("pseudopotentials on species with a core (cores/"), "{}", o.rows[0].model);
        assert!(o.rows[0].note.as_deref().unwrap().contains("species that have a core"));
        assert_eq!(o.blocked(), None);
        // another build's recipe: the same policy word, another hash
        p.ionic = "all/0123456789abcdef/shared0/0/stop2x0".into();
        assert_ne!(p.ionic, all_tag);
        let o = offer(Path::new("x.model"), &p, &s, &c).unwrap();
        assert!(o.blocked().unwrap().contains("different pseudopotential recipe"), "{:?}", o.blocked());
        assert!(o.edits.is_empty());
        // an unknown tag
        p.ionic = "weird/1".into();
        assert!(offer(Path::new("x.model"), &p, &s, &c).unwrap().blocked().unwrap().contains("not one this build knows"));
        // the policy is already the model's but nothing in the scene is pseudised under it
        let mut s2 = scene();
        s2.ionic = sim::IonicPolicy::CoresOnly;
        let mut p = prov_of(&s2);
        p.ionic = providers::pseudo::tag_of(sim::IonicPolicy::CoresOnly);
        let c = Context { pseudised: false, ..ctx(&ok) };
        let o = offer(Path::new("x.model"), &p, &s2, &c).unwrap();
        assert!(o.blocked().unwrap().contains("nothing here has a core"), "{:?}", o.blocked());
    }

    /// Final review M6 (the M7 leftovers): an `all/…` model offered in a run whose H would go through the dimer gate on
    /// the CPU says so on its row — about 40 min, before the click — and says nothing when the gate costs seconds (the
    /// disk cache, a cached table) or the scene has no H or He.
    #[test]
    fn the_ionic_row_names_a_cpu_dimer_gate_before_apply() {
        use providers::pseudo::GenerationCost as G;
        let s = scene(); // H₂ under CoresOnly: nothing pseudised
        let mut p = prov_of(&s);
        p.ionic = providers::pseudo::tag_of(sim::IonicPolicy::All);
        let cpu = |z: u8| if z == 1 { G::CpuGate } else { G::Seconds };
        let c = Context { generation_cost: &cpu, ..ctx(&ok) };
        let o = offer(Path::new("x.model"), &p, &s, &c).unwrap();
        assert_eq!(o.edits, vec![Edit::Ionic(sim::IonicPolicy::All)]);
        let note = o.rows[0].note.as_deref().expect("the CPU gate is said");
        assert_eq!(note, "applying generates the H pseudopotential on the CPU first — about 40 min (the dimer check; this machine has no GPU for it), and the first solve waits for it");
        assert_eq!(o.blocked(), None, "said, not refused: the user may want it");
        let fast = |_: u8| G::Seconds;
        let o = offer(Path::new("x.model"), &p, &s, &Context { generation_cost: &fast, ..ctx(&ok) }).unwrap();
        assert_eq!(o.rows[0].note, None, "replayed or cached: nothing to say");
        let mut he = s.clone();
        he.nuclei[1].z = 2;
        let both = |z: u8| if z <= 2 { G::CpuGate } else { G::Seconds };
        let o = offer(Path::new("x.model"), &p, &he, &Context { generation_cost: &both, ..ctx(&ok) }).unwrap();
        assert!(o.rows.iter().find(|r| r.name == "ionic potential").unwrap().note.as_deref().unwrap().contains("H and He pseudopotentials on the CPU first — about 80 min"));
    }

    #[test]
    fn the_gpu_memory_guard_and_an_invalid_grid_block_the_grid_row() {
        let s = scene();
        let mut p = prov_of(&s);
        p.grid_n = 193;
        p.h = 0.1;
        let guard = |t: &Scene| if t.grid_n > 150 { Err(format!("refused: {}³ needs too much", t.grid_n)) } else { Ok(()) };
        let o = offer(Path::new("x.model"), &p, &s, &ctx(&guard)).unwrap();
        assert_eq!(o.rows[0].name, "grid");
        assert_eq!(o.rows[0].blocked.as_deref(), Some("refused: 193³ needs too much"));
        // an even n is not a grid
        p.grid_n = 80;
        p.h = 0.15;
        let o = make(&p, &s).unwrap();
        assert!(o.rows[0].blocked.as_deref().unwrap().contains("not valid"), "{:?}", o.rows[0]);
        // the guard sees the scene the edits produce, not the current one
        let seen = std::cell::RefCell::new(Vec::new());
        let spy = |t: &Scene| {
            seen.borrow_mut().push((t.grid_n, t.charge));
            Ok(())
        };
        let mut p = prov_of(&s);
        p.grid_n = 49;
        p.h = 0.25;
        p.charge = 1;
        offer(Path::new("x.model"), &p, &s, &ctx(&spy)).unwrap();
        assert_eq!(*seen.borrow(), vec![(49, 1)]);
    }

    #[test]
    fn an_impossible_charge_or_multiplicity_is_named() {
        let s = scene();
        let mut p = prov_of(&s);
        p.charge = 3; // H2 has 2 electrons
        assert!(make(&p, &s).unwrap().blocked().unwrap().contains("negative electrons"));
        let mut p = prov_of(&s);
        p.multiplicity = Some(2); // 2 electrons cannot be a doublet
        assert!(make(&p, &s).unwrap().blocked().unwrap().contains("impossible for 2 electrons"));
        p.multiplicity = Some(3);
        assert_eq!(make(&p, &s).unwrap().blocked(), None);
        // together: the charge makes the doublet possible
        p.charge = 1;
        p.multiplicity = Some(2);
        assert_eq!(make(&p, &s).unwrap().blocked(), None);
    }

    #[test]
    fn isotopes_are_listed_and_applied_to_single_isotope_elements_only() {
        let mut s = scene();
        s.nuclei.push(nucleus(2, None, 3.0));
        let mut p = prov_of(&s);
        p.charge = 1; // something that makes a card
        p.masses = vec![(1, 2), (2, 4)];
        let o = make(&p, &s).unwrap();
        let r = o.rows.iter().find(|r| r.name == "isotopes").unwrap();
        assert!(!r.compared && r.blocked.is_none());
        assert_eq!((r.model.as_str(), r.scene.as_str()), ("²H ⁴He", "¹H ⁴He"));
        // H -> D on both hydrogens, He already 4
        assert_eq!(&o.edits[1..], &[Edit::Isotope { index: 0, a: 2 }, Edit::Isotope { index: 1, a: 2 }]);
        // an element the model mixed is left alone, said so
        p.masses = vec![(1, 1), (1, 2), (2, 4)];
        let o = make(&p, &s).unwrap();
        assert!(o.edits.iter().all(|e| !matches!(e, Edit::Isotope { .. })), "{:?}", o.edits);
        assert!(o.rows.iter().find(|r| r.name == "isotopes").unwrap().note.as_deref().unwrap().contains("mixed isotopes"));
        // an element the scene lacks
        p.masses = vec![(8, 16)];
        let o = make(&p, &s).unwrap();
        assert!(o.edits.iter().all(|e| !matches!(e, Edit::Isotope { .. })));
        assert!(o.rows.iter().find(|r| r.name == "isotopes").unwrap().note.as_deref().unwrap().contains("not in this scene"));
    }
}
