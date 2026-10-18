//! The process-wide pseudopotential cache (M7 Task 9; spec §5.5, `sim` bullet): tables are generated on first use
//! by `qm::radial::generate` on the calling thread (the worker or a side solver; seconds per element) and cached
//! by (Z, recipe_hash); filtered tables by (Z, recipe_hash, h). Refusals are cached too. Nothing is loaded from disk
//! or shipped (spec §3, rule 1): every table is made from this program's own all-electron atom.
//!
//! # What is decided here
//! - [`setup_for`]: a scene's [`qm::IonicSetup`] under its [`sim::IonicPolicy`] — `None` when nothing is pseudised
//!   (policy `None`; `CoresOnly` with only core-less or refused species), so those scenes run today's all-electron
//!   path bit for bit. An `IonicError` (refused under `All`, h coarser than a table's `h_req`, a failed filtered
//!   gate) is a `ProviderError::Invalid`, raised before any solve.
//! - [`provenance_tag`]: the *surface* a scene is solved on, for the surrogate's provenance and the references' key.
//!   The tag follows the physics, not the menu: H₂ under `CoresOnly` has nothing pseudised and carries `"none"`.
//! - [`gpu_tol_floor`] / [`effective_gpu_tol`]: the floor-aware GPU stop (spec §7 risk 6).
//!
//! # Generation notes
//! A table generated on first use leaves one note ("generated the O pseudopotential: …") in the cache; the next
//! [`setup_for`] — whichever scene it is for — hands it to that evaluation's `EvalStats::notes` and clears it.
//! The cache is behind one mutex held during generation: a second thread that needs a species being generated waits
//! for it (seconds, once per species per run).
use qm::ionic::{IonicPolicy as QmPolicy, IonicSetup};
use qm::radial::filter::{filtered, FilteredTable};
use qm::radial::{generate, has_core, ledger::recipe_hash as compute_recipe_hash, GenerationRecord, PseudoTable, Refusal};
use sim::{IonicPolicy, ProviderError, PseudoSummary, Scene};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The recipe hash of this build, computed once: `recipe_hash` hashes the whole ledger, and the cache is consulted on every
/// evaluation of the surrogate (the provenance tag), where microseconds matter.
fn recipe_hash() -> u64 {
    static HASH: OnceLock<u64> = OnceLock::new();
    *HASH.get_or_init(compute_recipe_hash)
}

/// The cache (module doc). Obtain it with [`cache`].
#[derive(Default)]
pub struct PseudoCache {
    tables: HashMap<(u8, u64), Result<Arc<PseudoTable>, Refusal>>,
    filtered: HashMap<(u8, u64, u64), Arc<FilteredTable>>,
    cores: HashMap<(u8, u64), Result<bool, Refusal>>,
    generated: u64,
    fresh: Vec<String>,
}

/// The one cache of the process.
pub fn cache() -> &'static Mutex<PseudoCache> {
    static CACHE: OnceLock<Mutex<PseudoCache>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn lock() -> std::sync::MutexGuard<'static, PseudoCache> {
    cache().lock().unwrap_or_else(|e| e.into_inner())
}

impl PseudoCache {
    /// Tables generated (or refused) by `qm::radial::generate` so far in this process.
    pub fn generated(&self) -> u64 {
        self.generated
    }

    /// The table of Z, generated on first use (a note for the next `setup_for` says so).
    pub(crate) fn table(&mut self, z: u8) -> Result<Arc<PseudoTable>, Refusal> {
        let key = (z, recipe_hash());
        if let Some(t) = self.tables.get(&key) {
            return t.clone();
        }
        let t0 = std::time::Instant::now();
        let t = generate(z).map(Arc::new);
        self.generated += 1;
        let sym = units::elements::symbol(z);
        match &t {
            Ok(t) => self.fresh.push(format!(
                "generated the {sym} pseudopotential: f = {}, {}, r_c ≤ {:.2} Bohr, h_req {} ({:.1} s)",
                t.f,
                t.record.local,
                t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max),
                t.record.h_req.map_or("none".into(), |h| format!("{h} Bohr")),
                t0.elapsed().as_secs_f64()
            )),
            Err(r) => self.fresh.push(format!("{sym}: {r}")),
        }
        self.tables.insert(key, t.clone());
        t
    }

    fn filtered(&mut self, z: u8, h: f64) -> Result<Arc<FilteredTable>, Refusal> {
        let key = (z, recipe_hash(), h.to_bits());
        if let Some(t) = self.filtered.get(&key) {
            return Ok(t.clone());
        }
        let ft = Arc::new(filtered(&*self.table(z)?, h));
        self.filtered.insert(key, ft.clone());
        Ok(ft)
    }

    fn has_core(&mut self, z: u8) -> Result<bool, Refusal> {
        self.cores.entry((z, recipe_hash())).or_insert_with(|| has_core(z)).clone()
    }

    /// Whether Z's table (or its refusal) is already in the cache — never generates.
    pub fn has_table(&self, z: u8) -> bool {
        self.tables.contains_key(&(z, recipe_hash()))
    }

    /// Whether `z` is pseudised under `policy`: `All` pseudises every species the recipe accepts; `CoresOnly` those
    /// with a core that it accepts. A refusal is an all-electron fallback under `CoresOnly` (an error under `All`,
    /// which `setup_for` reports) — so the answer for a refused species is `false` either way. Generates the table
    /// when the species needs one and none is cached (module doc).
    fn pseudised(&mut self, policy: IonicPolicy, z: u8) -> bool {
        let core = if policy == IonicPolicy::CoresOnly { self.has_core(z) } else { Ok(false) };
        wants_table(policy, &core) && self.table(z).is_ok()
    }
}

/// The one rule of which species `IonicSetup::build` hands a table to (spec §4.7): `All` every species, `CoresOnly`
/// those with a core (`core` = `has_core(z)`; a recipe refusal of the core test keeps the species all-electron),
/// `None` none. The tag, the memory guard and the chrome all go through it, so they cannot drift from each other.
fn wants_table(policy: IonicPolicy, core: &Result<bool, Refusal>) -> bool {
    match policy {
        IonicPolicy::None => false,
        IonicPolicy::All => true,
        IonicPolicy::CoresOnly => matches!(core, Ok(true)),
    }
}

/// `sim`'s policy as `qm`'s.
pub fn qm_policy(p: IonicPolicy) -> QmPolicy {
    match p {
        IonicPolicy::None => QmPolicy::None,
        IonicPolicy::CoresOnly => QmPolicy::CoresOnly,
        IonicPolicy::All => QmPolicy::All,
    }
}

/// The table every provider uses for Z — the one place a caller outside the providers reads `h_req`, so it
/// reads the same table the solves use (plan review M6: from Task 11 that is the dimer-gated table, never the
/// provisional one `qm::radial::generate` returns for Z ≤ dimer_z_max). Generated on first use.
pub fn table_for(z: u8) -> Result<Arc<PseudoTable>, Refusal> {
    lock().table(z)
}

/// `None` under `IonicPolicy::None` (`qm::System::new`: today's solver); otherwise the setup for the scene's
/// species at its h, or `None` when nothing is pseudised (CoresOnly with H/He or refused species only: today's
/// path, bit for bit). `IonicError` → `ProviderError::Invalid` (refused under `All`, h > h_req, a filtered gate),
/// before any solve. Generation and fallback notes go to `notes`.
pub fn setup_for(scene: &Scene, sigma_ae: f64, notes: &mut Vec<String>) -> Result<Option<Arc<IonicSetup>>, ProviderError> {
    setup_in(cache(), scene, sigma_ae, notes)
}

/// [`setup_for`] on a given cache (tests use a private one: no other thread's generation can touch it).
pub fn setup_in(cache: &Mutex<PseudoCache>, scene: &Scene, sigma_ae: f64, notes: &mut Vec<String>) -> Result<Option<Arc<IonicSetup>>, ProviderError> {
    if scene.ionic == IonicPolicy::None {
        return Ok(None);
    }
    let zs: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
    // two passes, so each closure borrows alone: the core test for every species (milliseconds), then the
    // filtered tables (generation on first use)
    let cores: HashMap<u8, Result<bool, Refusal>> = zs.iter().map(|&z| (z, c.has_core(z))).collect();
    let built = IonicSetup::build(qm_policy(scene.ionic), scene.cell.h, sigma_ae, &zs, &mut |z| cores[&z].clone(), &mut |z| c.filtered(z, scene.cell.h));
    notes.append(&mut c.fresh);
    let setup = built.map_err(|e| ProviderError::Invalid(e.to_string()))?;
    notes.extend(setup.notes.iter().cloned());
    Ok((!setup.pseudised().is_empty()).then(|| Arc::new(setup)))
}

/// The surface the teacher solves: `"none"` when nothing is pseudised (policy `None`; `CoresOnly` with only core-less
/// or refused species), else `"<policy word>/<recipe_hash:016x>/shared<τ>/<τ_c>/stop2x<floor>"`: the policy, the recipe
/// version, the occupation scheme the pseudo path solves with (`DegenerateShells`: τ = `qm::DEGENERACY_TAU` decides
/// the shared state and τ_c = `qm::scf::CANDIDATE_TAU` which levels are candidates — Task 6 review A.6, Task 9 physics
/// review SHOULD-3) and the GPU stop *rule* max(tol, 2·floor·N) ([`GPU_DRHO_FLOOR_PER_ELECTRON`]). The rule, not its
/// N-dependent value, is the compared identity: an Add or Remove must not split a model's surface (MUST-1).
pub fn provenance_tag(scene: &Scene) -> String {
    tag_in(cache(), scene)
}

/// [`provenance_tag`] on a given cache.
pub fn tag_in(cache: &Mutex<PseudoCache>, scene: &Scene) -> String {
    if scene.ionic == IonicPolicy::None || scene.nuclei.is_empty() {
        return "none".into();
    }
    let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
    if scene.nuclei.iter().any(|n| c.pseudised(scene.ionic, n.z)) {
        format!(
            "{}/{:016x}/shared{:e}/{:e}/stop2x{:e}",
            scene.ionic.word(),
            recipe_hash(),
            qm::DEGENERACY_TAU,
            qm::scf::CANDIDATE_TAU,
            GPU_DRHO_FLOOR_PER_ELECTRON
        )
    } else {
        "none".into()
    }
}

/// The f32 storage-rounding floor of the GPU SCF's Δρ (L1) per valence electron: measured 3.3e-8·N on pseudo-O
/// (N = 6, reaches 2e-7) and 4.6e-8·N on O₂ (N = 12, floors at 5.5e-7); 5e-8 is the upper end of the measured band
/// (spec §7 risk 6; Task 8a review). A solver constant, not a table value.
pub const GPU_DRHO_FLOOR_PER_ELECTRON: f64 = 5e-8;

/// The Δρ floor of an N-electron GPU solve (see [`GPU_DRHO_FLOOR_PER_ELECTRON`]).
pub fn gpu_tol_floor(n_valence: i64) -> f64 {
    GPU_DRHO_FLOOR_PER_ELECTRON * n_valence.max(0) as f64
}

/// `max(tol, 2·floor(N))`: the SCF tolerance the GPU can actually reach (as LOBPCG's 3·floor clamp does for the
/// eigensolver). A requested tolerance below ≈ 2× the floor costs steps or never converges. Applied only under an
/// ionic setup; the all-electron path is untouched.
pub fn effective_gpu_tol(tol: f64, n_valence: i64) -> f64 {
    tol.max(2.0 * gpu_tol_floor(n_valence))
}

/// What the GPU memory guard needs of a scene under an ionic policy, **without generating anything and without
/// waiting** — the guard is asked on the render thread (a preset, a quality level, a policy switch), where a
/// pseudopotential's seconds of generation must never run, and its answer must not depend on whether the worker
/// has generated a table yet. A species whose table is already generated contributes its exact valence charge and
/// ion extent; one whose table is not (or while the cache is busy generating) contributes the conservative bound:
/// all Z electrons and a generic ion ([`GENERIC_ION`]). `pseudised` is false when nothing is (policy `None`; `CoresOnly`
/// with only core-less or refused species), so those scenes are counted as today.
#[derive(Clone, Debug, PartialEq)]
pub struct FootprintInputs {
    pub pseudised: bool,
    pub n_valence: i64,
    pub ions: Vec<(f64, u64)>,
}

/// The largest table support R (Bohr) of an accepted species — the generic bound's recipe constant: r_floor = 3.5 Bohr
/// for Z_v = 6 at h = 0.35 (spec §5.1), above every accepted species' r_cmp (O 1.53, Ar 2.45).
pub const GENERIC_R_MAX: f64 = 3.5;

/// The extent the GPU's point lists use for a table, `γ_r·R + 8h` with R the table's support (module doc of
/// `qm::radial::filter`): from a cached *filtered* table at this h the exact max(projector radius, R_SR, ρ_pc radius); from
/// an unfiltered table the support bounded by max(r_cmp, r_c, r_nlc); from nothing the recipe constant
/// [`GENERIC_R_MAX`]. With four projector rows (an s and a p channel) in the last case.
fn ion_extent(c: &PseudoCache, z: u8, h: f64) -> (f64, u64) {
    let ledger = qm::radial::ledger::LEDGER;
    let reach = |r: f64| ledger.gamma_r * r + ledger.numerics.kspl_step_over_h * h;
    let hash = recipe_hash();
    if let Some(f) = c.filtered.get(&(z, hash, h.to_bits())) {
        let rows = f.projectors.iter().map(|p| 2 * p.l as u64 + 1).sum();
        let r = f.projectors.iter().map(|p| p.radius).fold(0.0, f64::max).max(f.r_sr).max(f.pc_radius);
        return (r, rows);
    }
    match c.tables.get(&(z, hash)) {
        Some(Ok(t)) => {
            let rows = t.kb.projectors.iter().map(|p| 2 * p.l as u64 + 1).sum();
            let r = t.channels.iter().map(|ch| ch.r_c).fold(0.0, f64::max).max(t.kb.r_cmp).max(t.partial_core.r_nlc.unwrap_or(0.0));
            (reach(r), rows)
        }
        _ => (reach(GENERIC_R_MAX), 4),
    }
}

/// [`FootprintInputs`] of `scene` (see there).
pub fn footprint_inputs(scene: &Scene) -> FootprintInputs {
    footprint_in(cache(), scene)
}

/// [`footprint_inputs`] on a given cache (tests use a private one).
pub fn footprint_in(cache: &Mutex<PseudoCache>, scene: &Scene) -> FootprintInputs {
    let protons: i64 = scene.nuclei.iter().map(|n| n.z as i64).sum();
    let all_electron = FootprintInputs { pseudised: false, n_valence: protons - scene.charge as i64, ions: Vec::new() };
    if scene.ionic == IonicPolicy::None || scene.nuclei.is_empty() {
        return all_electron;
    }
    let h = scene.cell.h;
    let Ok(mut c) = cache.try_lock() else {
        let generic = (qm::radial::ledger::LEDGER.gamma_r * GENERIC_R_MAX + qm::radial::ledger::LEDGER.numerics.kspl_step_over_h * h, 4);
        return FootprintInputs { pseudised: true, n_valence: protons - scene.charge as i64, ions: vec![generic; scene.nuclei.len()] };
    };
    let hash = recipe_hash();
    let (mut any, mut n_valence, mut ions) = (false, 0i64, Vec::new());
    for n in &scene.nuclei {
        let z = n.z;
        // the core test is a radial atom (milliseconds, cached): cheap enough for the render thread
        let core = if scene.ionic == IonicPolicy::CoresOnly { c.has_core(z) } else { Ok(false) };
        if !wants_table(scene.ionic, &core) {
            n_valence += z as i64;
            continue;
        }
        match c.tables.get(&(z, hash)) {
            Some(Ok(t)) => {
                any = true;
                n_valence += t.z_v.round() as i64;
                ions.push(ion_extent(&c, z, h));
            }
            // refused: all-electron under CoresOnly (an error under All, which the provider reports itself)
            Some(Err(_)) => n_valence += z as i64,
            // not generated yet: the bound
            None => {
                any = true;
                n_valence += z as i64;
                ions.push(ion_extent(&c, z, h));
            }
        }
    }
    if any { FootprintInputs { pseudised: true, n_valence: n_valence - scene.charge as i64, ions } } else { all_electron }
}

/// What the generator recorded for each pseudised species of `setup`, for the badge, the Selection tab and the
/// Quality menu (one entry per Z, ascending).
pub fn summaries(setup: &IonicSetup) -> Vec<PseudoSummary> {
    setup
        .per_z
        .iter()
        .filter_map(|(z, i)| match i {
            qm::ionic::Ionic::Pseudo(t) => Some(summary_of(*z, t.z_v, t.r_core, &t.record)),
            _ => None,
        })
        .collect()
}

fn summary_of(z: u8, z_v: f64, r_core: f64, r: &GenerationRecord) -> PseudoSummary {
    PseudoSummary {
        z,
        z_v,
        f: r.f,
        local: r.local.clone(),
        r_c_max: r.r_c.iter().map(|(_, r)| *r).fold(0.0, f64::max),
        h_req: r.h_req,
        h_eb: r.h_eb,
        r_core,
        record: record_text(r),
    }
}

/// The summaries of the species of `scene` whose tables are *already generated* and that the scene's policy
/// pseudises — never generating, never waiting (`try_lock`: a table being generated returns nothing). What the
/// chrome can say about a scene before (or without) a successful solve: the Quality menu marks the grids a
/// species' `h_req` forbids even when the scene was refused for one.
pub fn cached_summaries(scene: &Scene) -> Vec<PseudoSummary> {
    cached_in(cache(), scene)
}

/// What the process has already learned about Z's pseudopotential: the generated table or the refusal, `None` when
/// nothing has been generated for it yet (or the cache is busy generating — `try_lock`). Never generates, never waits
/// (the periodic table's tooltips read it on the render thread).
pub fn cached_outcome(z: u8) -> Option<Result<Arc<PseudoTable>, Refusal>> {
    let c = cache().try_lock().ok()?;
    c.tables.get(&(z, recipe_hash())).cloned()
}

/// [`cached_summaries`] on a given cache (tests use a private one).
pub fn cached_in(cache: &Mutex<PseudoCache>, scene: &Scene) -> Vec<PseudoSummary> {
    if scene.ionic == IonicPolicy::None {
        return Vec::new();
    }
    let Ok(c) = cache.try_lock() else { return Vec::new() };
    let hash = recipe_hash();
    let mut zs: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    zs.sort_unstable();
    zs.dedup();
    zs.into_iter()
        .filter_map(|z| {
            let t = c.tables.get(&(z, hash))?.as_ref().ok()?;
            let core = c.cores.get(&(z, hash)).cloned().unwrap_or(Ok(false));
            wants_table(scene.ionic, &core).then(|| summary_of(z, t.z_v, t.record.r_core, &t.record))
        })
        .collect()
}

/// The generation record as the Selection tab's text: what the recipe chose and measured for this element.
fn record_text(r: &GenerationRecord) -> String {
    let per_l = |v: &[(u8, f64)], d: usize| v.iter().map(|(l, x)| format!("{}: {x:.d$}", ["s", "p", "d", "f"].get(*l as usize).copied().unwrap_or("?"))).collect::<Vec<_>>().join(", ");
    let opt = |x: Option<f64>| x.map_or("none".to_string(), |h| format!("{h:.2} Bohr"));
    let mut s = format!(
        "recipe {:016x} · Z = {}, Z_v = {}, {} core electrons\nsoftness f = {}, local channel: {}\nr_c (Bohr): {}\nE_KB (Ha): {}\nGSS (Ha): {}\n",
        r.recipe_hash, r.z, r.z_v, r.n_core, r.f, r.local, per_l(&r.r_c, 3), per_l(&r.e_kb, 4), per_l(&r.gss, 4)
    );
    s += &format!(
        "r_nlc = {}, r_cmp = {:.3} Bohr, r_core = {:.3} Bohr\nh_req = {} (energies certified at and below)\nh_eb = {} (egg-box force certified at and below)\ndimer gate: {}\ngenerated in {:.1} s\n",
        r.r_nlc.map_or("—".to_string(), |x| format!("{x:.3} Bohr")), r.r_cmp, r.r_core, opt(r.h_req), opt(r.h_eb), r.dimer, r.seconds
    );
    if !r.filter.is_empty() {
        s += "filter ladder (h, |ΔE| mHa, max|Δε| mHa, table egg-box Ha/Bohr):\n";
        for f in &r.filter {
            s += &format!("  {:.2}  {:.3}  {:.3}  {:.1e}  {}\n", f.h, f.de.abs() * 1e3, f.max_eig * 1e3, f.f_atom_nl(), if f.passes() { "pass" } else { "fail" });
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::{Cell, Nucleus};

    fn scene(z: &[u8], policy: IonicPolicy, n: usize, h: f64) -> Scene {
        Scene {
            nuclei: z.iter().enumerate().map(|(i, &z)| Nucleus::new(z, [i as f64 * 2.0, 0.0, 0.0])).collect(),
            cell: Cell { n, h },
            charge: 0,
            multiplicity: None,
            sigma: None,
            external_field: [0.0; 3],
            ionic: policy,
        }
    }

    /// The core-overlap warning prefix `sim` carries is `qm`'s text (the driver and the surrogate match on it).
    #[test]
    fn core_overlap_warning_text_is_qms() {
        assert_eq!(sim::CORE_OVERLAP_WARNING, qm::ionic::CORE_OVERLAP_WARNING);
    }

    /// The hand-copied measurements of the chrome and the README (`app::ui::pseudo::EGG_BOX_RECORD`, the per-element
    /// certified-grid list, the h_req / h_eb numbers) were taken at this recipe version: a bump of the ledger must
    /// go with a review of them (task-9 review M7).
    #[test]
    fn the_copied_egg_box_figures_belong_to_this_recipe_version() {
        assert_eq!(qm::radial::ledger::LEDGER.version, "m7-tm-kb-nlcc-4", "recipe changed: re-read crates/app/src/ui/pseudo.rs EGG_BOX_RECORD and README.md's Pseudopotentials section against references.md, then update this pin");
    }

    /// Spec §7 risk 6: the floor-aware tolerance is the requested one for small N and rises above N = 10.
    #[test]
    fn effective_gpu_tol_is_the_request_until_the_floor_reaches_it() {
        assert_eq!(effective_gpu_tol(1e-6, 6), 1e-6, "pseudo-O: 2·floor = 6e-7 < 1e-6");
        assert_eq!(effective_gpu_tol(1e-6, 10), 1e-6, "N = 10 is the break-even");
        assert!((effective_gpu_tol(1e-6, 12) - 1.2e-6).abs() < 1e-18, "O₂: 2·floor = 1.2e-6");
        assert_eq!(effective_gpu_tol(1e-5, 12), 1e-5, "the interactive tolerance is above the floor");
        assert_eq!(effective_gpu_tol(1e-6, -3), 1e-6);
    }

    /// H₂ under CoresOnly has nothing pseudised: the tag is "none" (the all-electron surface) and `setup_for` is
    /// `None`; under `None` the cache is never touched.
    #[test]
    fn nothing_pseudised_is_the_all_electron_surface() {
        // a private cache: no other test thread can generate into it or hold its lock, so the assertions always run
        let private = Mutex::new(PseudoCache::default());
        let h2 = scene(&[1, 1], IonicPolicy::CoresOnly, 33, 0.25);
        assert_eq!(tag_in(&private, &h2), "none");
        let mut notes = Vec::new();
        assert!(setup_in(&private, &h2, 0.25, &mut notes).unwrap().is_none());
        assert_eq!(tag_in(&private, &scene(&[8], IonicPolicy::None, 33, 0.25)), "none");
        assert!(setup_in(&private, &scene(&[8], IonicPolicy::None, 33, 0.25), 0.25, &mut notes).unwrap().is_none());
        assert!(notes.is_empty());
        let c = private.lock().unwrap();
        assert_eq!(c.generated(), 0, "no table was generated for H₂ and for the None policy");
        assert!(!c.has_table(1) && !c.has_table(8));
    }
}
