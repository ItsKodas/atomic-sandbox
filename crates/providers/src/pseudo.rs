//! The process-wide pseudopotential cache (M7 Task 9; spec §5.5, `sim` bullet): tables are generated on first use
//! by `qm::radial::generate_with` on the calling thread (the worker or a side solver; seconds per element, plus the
//! dimer gate for H and He) and cached by (Z, recipe_hash); filtered tables by (Z, recipe_hash, h). Refusals are cached
//! too. Nothing is loaded from disk or shipped (spec §3, rule 1): every table is made from this program's own
//! all-electron atom.
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
//! for it (seconds, once per species per run; for H and He the dimer gate's ≈ 1 min on the GPU).
//!
//! # The dimer gate on the grid (M7 Task 11; spec §4.4, Spec question 9 ruling; `qm::radial::recipe` "The dimer gate")
//! The cache always generates through [`GridDimerGate`], so the H and He tables it hands out are never provisional
//! (plan review M6: one hash, one potential). Atomic units (Hartree, Bohr, Ha/Bohr; ω in E_h/ħ).
//! - **Grid.** The finest ladder spacing, h = `LEDGER.h_ladder` last = 0.15 Bohr, on a 12 Bohr cube (81³, a grid point at
//!   the origin): the app's default (Accurate) and the σ → 0 oracle's grid (`validate::oracles` `h2_bond_length`).
//! - **Geometry.** Two nuclei of charge Z at (∓R/2, 0, 0) — inversion-symmetric on the grid, so F₀ = −F₁ up to f32
//!   noise; the bond force is F(R) = ½(F₁,x − F₀,x) = −dE/dR (positive = repulsive).
//! - **Pseudised dimer.** The table filtered at h, through `IonicSetup::from_tables` (the candidate has no h_req yet: T8
//!   runs on the accepted entry), `DegenerateShells` occupations (the providers' choice under a pseudo policy), cold
//!   solves, tol 1e-6 (GPU: the floor-aware `effective_gpu_tol`), eig_tol 1e-7.
//! - **All-electron dimer.** At each R, F(R; σ) at σ = h·{1.0, 1.2, 1.4, 1.6} (the oracle's ladder, `validate::sigma_ladder`;
//!   restated, not imported — providers do not depend on validate), cold solves, tol 1e-6, eig_tol 1e-7, integer aufbau,
//!   then F₀(R) = `qm::extrapolate_sigma` (the derivative of the energy ansatz a + bσ² + cσ³ + dσ⁴ has the same σ
//!   dependence; Zσ_max = 0.24 for H is inside its asymptotic regime). The extrapolation weights sum to |w|₁ ≈ 88, so a
//!   force noise u per solve is ≈ 88u in F₀: the GPU's H₂ force differs from the CPU's by ≤ 3e-6 at tol 1e-6
//!   (references.md "GPU vs CPU at fixed σ"), ≈ 3e-4 in F₀ — δR_e ≈ 3e-4/k ≈ 1e-3 Bohr against the 0.02 gate.
//! - **R_e.** Bisection of the sign of F in `LEDGER.dimer_bracket`·r_peak (r_peak: the outermost occupied valence maximum of
//!   the AE reference) until the bracket is ≤ `dimer_re`/10, then the root by linear interpolation in the last bracket
//!   (the plan took the midpoint, ≤ 1e-3 Bohr off; interpolation is better and the same on both sides). F(lo) > 0 > F(hi)
//!   is required, else there is no force zero in the bracket: for the AE dimer the gate is not applicable (with both
//!   dimers' F at both ends recorded, plan review Q9(ii)); for the pseudised one the entry fails.
//! - **k and ω.** k = −[F(R_e + h) − F(R_e − h)]/(2h), ω = √(k/μ), μ = m/2 with m the most abundant isotope's mass
//!   (`units::isotope_mass_amu`; the one per-element datum allowed). Bias (Q9(iii)): for a Morse curve
//!   k_FD = k(1 + 7a²h²/6), ≈ +2.8 % in k, +1.4 % in ω for H₂ at h = 0.15; it cancels inside the gate (the same procedure
//!   on both sides), not against a curve-fitted ω.
//! - **Cost** (reported): ≈ 15 R points per dimer: 60 AE solves (cached per Z for the gate's lifetime, so later f's
//!   cost only the 15 PP solves) — ≈ 1 min on the RX 7900 XTX; on the CPU fallback (no adapter) ≈ 40 min per element.
//! - **Backend.** Process-wide ([`set_dimer_backend`]): the app sets the window's GPU, the validate suites theirs; the
//!   default is the CPU.
use qm::ionic::{IonicPolicy as QmPolicy, IonicSetup};
use qm::radial::filter::{filtered, FilteredTable};
use qm::radial::{generate_with, has_core, ledger::recipe_hash as compute_recipe_hash, DimerError, DimerGate, DimerPair, GenerationRecord, PseudoTable, Refusal};
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
        // the gate runs only for Z ≤ LEDGER.dimer_z_max: the ladder decides that, not the cache
        let mut gate = GridDimerGate::new(dimer_backend());
        let t = generate_with(z, Some(&mut gate)).map(Arc::new);
        self.generated += 1;
        let sym = units::elements::symbol(z);
        match &t {
            Ok(t) => self.fresh.push(format!(
                "generated the {sym} pseudopotential: f = {}, {}, r_c ≤ {:.2} Bohr, h_req {} ({:.1} s){}",
                t.f,
                t.record.local,
                t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max),
                t.record.h_req.map_or("none".into(), |h| format!("{h} Bohr")),
                t0.elapsed().as_secs_f64(),
                if gate.solves > 0 { format!("; dimer gate: {} ({} solves)", t.record.dimer, gate.solves) } else { String::new() }
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
/// or refused species), else `"<policy word>/<surface_hash:016x>/shared<τ>/<τ_c>/stop2x<floor>"`: the policy, the identity
/// of the tables the policy can hand out ([`surface_hash`]: since Task 11 not the whole recipe — a `CoresOnly` model from
/// before the dimer gate keeps its provenance, an `All` one does not), the occupation scheme the pseudo path solves with (`DegenerateShells`: τ = `qm::DEGENERACY_TAU` decides
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
            surface_hash(&mut c, scene.ionic),
            qm::DEGENERACY_TAU,
            qm::scf::CANDIDATE_TAU,
            GPU_DRHO_FLOOR_PER_ELECTRON
        )
    } else {
        "none".into()
    }
}

/// The identity of the tables `policy` can pseudise, for the provenance tag (M7 Task 11; the controller's requirement:
/// a provenance is invalidated exactly when a table its surface can use changes). The dimer gate moves only the tables of
/// Z ≤ `dimer_z_max` (`qm::radial::ledger` "Two identities"), so: `qm::radial::ledger::recipe_hash` (ledger + gate) when the
/// policy would pseudise any such species (`All`), else `core_recipe_hash` (the ledger alone — `CoresOnly`, because H and He
/// have no core, `has_core` decides). Per policy, never per scene: the tag must not depend on which species a scene holds,
/// or an Add or Remove would split a model's surface (Task 9 review MUST-1) — so an `All` model with no H or He in it is
/// refused across the gate's introduction too (over-invalidation on the safe side; it was made with a table set that
/// included the ungated H).
pub(crate) fn surface_hash(c: &mut PseudoCache, policy: IonicPolicy) -> u64 {
    let ledger = qm::radial::ledger::LEDGER;
    let gated = (1..=ledger.dimer_z_max).any(|z| {
        let core = if policy == IonicPolicy::CoresOnly { c.has_core(z) } else { Ok(false) };
        wants_table(policy, &core)
    });
    if gated { recipe_hash() } else { qm::radial::ledger::core_recipe_hash() }
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

// ---------------------------------------------------------------- the dimer gate (module doc)

/// Where the dimer gate solves (module doc, "Backend").
#[derive(Clone)]
pub enum DimerBackend { Cpu, Gpu(gpu::GpuContext) }

static BACKEND: Mutex<Option<DimerBackend>> = Mutex::new(None);

/// The process-wide dimer-gate backend (default: CPU). The app sets the window's GPU; the validate suites theirs.
pub fn set_dimer_backend(b: DimerBackend) {
    *BACKEND.lock().unwrap_or_else(|e| e.into_inner()) = Some(b);
}

/// The backend [`PseudoCache`] generates H and He with.
pub fn dimer_backend() -> DimerBackend {
    BACKEND.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or(DimerBackend::Cpu)
}

/// Whether a GPU backend is set (the chrome's cost note).
pub fn dimer_backend_is_gpu() -> bool {
    matches!(dimer_backend(), DimerBackend::Gpu(_))
}

/// The gate's grid (module doc): the finest ladder h on a 12 Bohr cube.
fn finest_grid() -> (qm::Grid, f64) {
    let h = *qm::radial::ledger::LEDGER.h_ladder.last().expect("a ladder");
    (qm::Grid::cube(2 * (6.0 / h).round() as usize + 1, h), h)
}

fn dimer_nuclei(z: u8, r: f64) -> Vec<qm::Nucleus> {
    vec![qm::Nucleus { pos: [-0.5 * r, 0.0, 0.0], z }, qm::Nucleus { pos: [0.5 * r, 0.0, 0.0], z }]
}

/// The bond-force solver of one gate: the backend, its grid and (GPU) one solver reused cold; counts its solves.
struct BondForce {
    backend: DimerBackend,
    grid: qm::Grid,
    gpu: Option<qm_gpu::GpuSolver>,
    solves: usize,
    /// The all-electron F(R; σ) of the last `all_electron` call, for the gate's log.
    last_sigma: String,
}

impl BondForce {
    fn new(backend: DimerBackend) -> BondForce {
        BondForce { backend, grid: finest_grid().0, gpu: None, solves: 0, last_sigma: String::new() }
    }

    /// ½(F₁,x − F₀,x) of the dimer `sys` (nucleus 1 at +R/2) from a cold solve at `settings`.
    fn at(&mut self, sys: &qm::System, settings: &qm::ScfSettings) -> Result<f64, String> {
        let grid = &self.grid;
        let f = match &self.backend {
            DimerBackend::Cpu => {
                let (st, _) = qm::scf(grid, sys, settings, None).map_err(|e| e.to_string())?;
                qm::forces(grid, &st, &sys.nuclei, settings.sigma).map_err(|e| e.to_string())?
            }
            DimerBackend::Gpu(ctx) => {
                let solver = self.gpu.get_or_insert_with(|| qm_gpu::GpuSolver::new(ctx.clone(), grid));
                let mut s = *settings;
                if sys.ionic.is_some() {
                    s.tol = effective_gpu_tol(s.tol, (sys.n_up + sys.n_dn) as i64);
                }
                solver.solve(sys, &s, false).map_err(|e| e.to_string())?;
                let st = solver.state().ok_or("the GPU solve left no state")?;
                solver.forces(st, &sys.nuclei, s.sigma).map_err(|e| e.to_string())?
            }
        };
        self.solves += 1;
        Ok(0.5 * (f[1][0] - f[0][0]))
    }

    /// The pseudised dimer's bond force at R (module doc, "Pseudised dimer").
    fn pseudo(&mut self, z: u8, setup: &Arc<IonicSetup>, r: f64) -> Result<f64, String> {
        let mut s = qm::ScfSettings::for_grid(&self.grid);
        s.eig_tol = 1e-7;
        s.occupations = qm::Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        self.at(&qm::System::with_ionic(dimer_nuclei(z, r), 0, None, setup.clone()), &s)
    }

    /// The all-electron dimer's bond force at R, extrapolated σ → 0 (module doc, "All-electron dimer").
    fn all_electron(&mut self, z: u8, r: f64) -> Result<f64, String> {
        let h = self.grid.h;
        let sigmas = [1.0, 1.2, 1.4, 1.6].map(|f| f * h);
        let sys = qm::System::new(dimer_nuclei(z, r), 0, None);
        let mut fs = [0.0; 4];
        for (k, &sigma) in sigmas.iter().enumerate() {
            let mut s = qm::ScfSettings::for_grid(&self.grid);
            s.sigma = sigma;
            s.eig_tol = 1e-7;
            fs[k] = self.at(&sys, &s)?;
        }
        self.last_sigma = format!("F(σ = {}) = {}", sigmas.map(|x| format!("{x:.3}")).join("/"), fs.map(|x| format!("{x:+.6e}")).join("/"));
        Ok(qm::extrapolate_sigma(&sigmas, &fs))
    }
}

/// The pseudised table at the gate's h, unchecked (the candidate has no h_req yet).
fn pseudo_setup(t: &PseudoTable, h: f64) -> Arc<IonicSetup> {
    let ft = Arc::new(filtered(t, h));
    Arc::new(IonicSetup::from_tables(QmPolicy::All, h, h, std::collections::BTreeMap::from([(t.z, qm::ionic::Ionic::Pseudo(ft))])))
}

/// The bracket `LEDGER.dimer_bracket`·r_peak (module doc, "R_e").
fn bracket(t: &PseudoTable) -> (f64, f64) {
    let b = qm::radial::ledger::LEDGER.dimer_bracket;
    let r_peak = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.r_peak).fold(0.0, f64::max);
    (b[0] * r_peak, b[1] * r_peak)
}

/// R_e by bisection of the sign of F in [lo, hi] to `dimer_re`/10 and linear interpolation in the last bracket, then
/// k = −[F(R_e + h) − F(R_e − h)]/(2h) and ω = √(k/μ), μ = m/2 (module doc). `NotApplicable`: F is not + at lo and − at
/// hi; `Failed`: a solve failed or k ≤ 0. `log` gets one line per evaluation.
fn locate(z: u8, lo: f64, hi: f64, h: f64, log: &mut Vec<String>, force: &mut dyn FnMut(f64) -> Result<f64, String>) -> Result<(f64, f64), DimerError> {
    let mut f = |r: f64, log: &mut Vec<String>| -> Result<f64, DimerError> {
        let v = force(r).map_err(|e| DimerError::Failed(format!("R = {r:.4} Bohr: {e}")))?;
        log.push(format!("R {r:.5} Bohr: F {v:+.6e} Ha/Bohr"));
        Ok(v)
    };
    let (mut a, mut b) = (lo, hi);
    let (mut fa, mut fb) = (f(a, log)?, f(b, log)?);
    if !(fa > 0.0 && fb < 0.0) {
        return Err(DimerError::NotApplicable(format!("no force zero in [{lo:.3}, {hi:.3}] Bohr: F({lo:.3}) {fa:+.3e}, F({hi:.3}) {fb:+.3e} Ha/Bohr")));
    }
    while b - a > qm::radial::ledger::LEDGER.dimer_re / 10.0 {
        let m = 0.5 * (a + b);
        let fm = f(m, log)?;
        if fm > 0.0 { (a, fa) = (m, fm) } else { (b, fb) = (m, fm) }
    }
    let r_e = a + (b - a) * fa / (fa - fb);
    let k = -(f(r_e + h, log)? - f(r_e - h, log)?) / (2.0 * h);
    if !(k > 0.0) {
        return Err(DimerError::Failed(format!("k = {k:.3e} Ha/Bohr² ≤ 0 at R_e {r_e:.4} Bohr")));
    }
    let m = units::amu_to_au_mass(units::isotope_mass_amu(z).ok_or_else(|| DimerError::Failed(format!("no isotope mass for Z = {z}")))?);
    Ok((r_e, (k / (0.5 * m)).sqrt()))
}

/// The pseudised homonuclear dimer of `t` at the gate's grid (module doc): (R_e Bohr, ω a.u.).
pub fn pseudo_dimer(t: &PseudoTable, backend: &DimerBackend) -> Result<(f64, f64), String> {
    let mut log = Vec::new();
    pseudo_located(t, &mut BondForce::new(backend.clone()), &mut log).map_err(|e| format!("{e:?}"))
}

fn pseudo_located(t: &PseudoTable, solver: &mut BondForce, log: &mut Vec<String>) -> Result<(f64, f64), DimerError> {
    let h = solver.grid.h;
    let setup = pseudo_setup(t, h);
    let (lo, hi) = bracket(t);
    locate(t.z, lo, hi, h, log, &mut |r| solver.pseudo(t.z, &setup, r))
}

/// The all-electron homonuclear dimer of Z at σ → 0 on the gate's grid (module doc): (R_e Bohr, ω a.u.), or why not.
/// `t` gives the bracket (its AE reference's r_peak).
pub fn all_electron_dimer(t: &PseudoTable, backend: &DimerBackend) -> Result<(f64, f64), String> {
    let mut log = Vec::new();
    let mut solver = BondForce::new(backend.clone());
    let (lo, hi) = bracket(t);
    let h = solver.grid.h;
    locate(t.z, lo, hi, h, &mut log, &mut |r| solver.all_electron(t.z, r)).map_err(|e| format!("{e:?}"))
}

/// The grid dimer gate (module doc): `qm::radial::DimerGate` for `generate_with`. The all-electron dimer is computed
/// once per Z for the gate's lifetime (it does not depend on f). `log` holds every bond-force evaluation and verdict,
/// `solves` counts the SCF solves (reported).
pub struct GridDimerGate {
    pub backend: DimerBackend,
    ae_cache: HashMap<u8, Result<(f64, f64), DimerError>>,
    solver: Option<BondForce>,
    pub log: Vec<String>,
    pub solves: usize,
    pub seconds: f64,
}

impl GridDimerGate {
    pub fn new(backend: DimerBackend) -> GridDimerGate {
        GridDimerGate { backend, ae_cache: HashMap::new(), solver: None, log: Vec::new(), solves: 0, seconds: 0.0 }
    }
}

impl DimerGate for GridDimerGate {
    fn dimers(&mut self, t: &PseudoTable) -> Result<DimerPair, DimerError> {
        let t0 = std::time::Instant::now();
        let backend = self.backend.clone();
        let solver = self.solver.get_or_insert_with(|| BondForce::new(backend));
        let before = solver.solves;
        let h = solver.grid.h;
        let (lo, hi) = bracket(t);
        let sym = units::elements::symbol(t.z);
        let log = &mut self.log;
        let ae = self.ae_cache.entry(t.z).or_insert_with(|| {
            log.push(format!("{sym}₂ all-electron σ → 0 at h = {h} (σ = h·{{1, 1.2, 1.4, 1.6}}), bracket [{lo:.4}, {hi:.4}] Bohr:"));
            let mut trace = Vec::new();
            let out = locate(t.z, lo, hi, h, log, &mut |r| {
                let f0 = solver.all_electron(t.z, r);
                trace.push(format!("  R {r:.5}: {} → F₀ {}", solver.last_sigma, f0.as_ref().map_or_else(|e| e.clone(), |f| format!("{f:+.6e}"))));
                f0
            });
            log.extend(trace);
            out
        }).clone();
        log.push(format!("{sym}₂ pseudised, f = {}, {:?}:", t.f, t.local));
        let out = match ae {
            Ok(ae) => match pseudo_located(t, solver, log) {
                Ok(pp) => Ok(DimerPair { pp, ae }),
                Err(DimerError::NotApplicable(why)) => Err(DimerError::PseudoUnbound(why)),
                Err(e) => Err(e),
            },
            // not applicable: record the pseudised dimer's F at both bracket ends too (plan review Q9(ii), report only)
            Err(DimerError::NotApplicable(why)) => {
                let setup = pseudo_setup(t, h);
                let mut pp = |r: f64| solver.pseudo(t.z, &setup, r).map_or_else(|e| format!("error: {e}"), |f| format!("{f:+.3e}"));
                let (p_lo, p_hi) = (pp(lo), pp(hi));
                Err(DimerError::NotApplicable(format!("AE F (σ → 0): {why}; PP F({lo:.3}) {p_lo}, PP F({hi:.3}) {p_hi} Ha/Bohr")))
            }
            Err(e) => Err(e),
        };
        log.push(format!("  → {}", match &out { Ok(p) => format!("{p:?}"), Err(e) => format!("{e:?}") }));
        self.solves += solver.solves - before;
        self.seconds += t0.elapsed().as_secs_f64();
        out
    }
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
        // the dimer gate (Task 11) moved only H and He, through `DIMER_GATE`, not the version: the O, N, C figures stand
        assert_eq!(qm::radial::ledger::LEDGER.version, "m7-tm-kb-nlcc-4", "recipe changed: re-read crates/app/src/ui/pseudo.rs EGG_BOX_RECORD and README.md's Pseudopotentials section against references.md, then update this pin");
    }

    /// Spec §8 row 11 (§4.4, Spec question 9): the dimer gate runs inside the ladder for H and is not applicable for He,
    /// and its verdict is in the generation record; the gated tables are usable (plan review M6). The not-applicable
    /// record carries the PP and AE bond forces at both bracket ends (Q9(ii)). The AE side is reported against the σ → 0
    /// oracle (`validate::oracles` `h2_bond_length`, R_e 1.4511 Bohr on the CPU at this grid).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (GPU dimer solves); cargo test -p providers --release")]
    fn dimer_gate_runs_for_hydrogen_and_is_not_applicable_for_helium() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let mut gate = GridDimerGate::new(DimerBackend::Gpu(ctx));
        let t0 = std::time::Instant::now();
        let h = qm::radial::generate_with(1, Some(&mut gate)).expect("H generates");
        let t_h = t0.elapsed().as_secs_f64();
        let he = qm::radial::generate_with(2, Some(&mut gate)).expect("He generates");
        eprintln!("{}", gate.log.join("\n"));
        eprintln!("H: f = {}, {}, h_req {:?}, h_eb {:?}; dimer gate: {}\nHe: f = {}, {}, h_req {:?}; dimer gate: {}\n{} solves, H {t_h:.0} s, total {:.0} s (gate {:.0} s)",
                  h.f, h.record.local, h.record.h_req, h.record.h_eb, h.record.dimer, he.f, he.record.local, he.record.h_req, he.record.dimer,
                  gate.solves, t0.elapsed().as_secs_f64(), gate.seconds);
        assert!(h.record.dimer.starts_with("pass"), "{}", h.record.dimer);
        assert!(he.record.dimer.starts_with("not applicable"), "{}", he.record.dimer);
        assert!(he.record.dimer.contains("PP F") && he.record.dimer.contains("AE F"), "{}", he.record.dimer);
        assert!(!h.provisional && !he.provisional, "gated tables are usable");
    }

    /// M7 Task 11 (the controller's requirement): the provenance is invalidated exactly when a table its surface can use
    /// changes. Switching the dimer gate on changed H and He only, so a `CoresOnly` surface (O pseudised, H all-electron)
    /// keeps the tag it had at `m7-tm-kb-nlcc-4` — the exact string a -4 model's sidecar holds, with the -4 recipe hash
    /// 0xeef0a918057a9fbe (references.md's records block before Task 11) — while an `All` surface's tag moves (it may hold the
    /// gated H). The `All` side is checked on `surface_hash` (no generation: the gated H costs the GPU dimer gate).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
    fn cores_only_provenance_survives_the_dimer_gate_and_all_does_not() {
        const RECIPE_HASH_M7_4: u64 = 0xeef0_a918_057a_9fbe;
        let private = Mutex::new(PseudoCache::default());
        let water = scene(&[8, 1, 1], IonicPolicy::CoresOnly, 33, 0.2);
        let tag = tag_in(&private, &water);
        let minus_4 = format!("cores/{RECIPE_HASH_M7_4:016x}/shared{:e}/{:e}/stop2x{:e}", qm::DEGENERACY_TAU, qm::scf::CANDIDATE_TAU, GPU_DRHO_FLOOR_PER_ELECTRON);
        assert_eq!(tag, minus_4, "a CoresOnly water model from -4 must still load");
        assert_eq!(tag_in(&private, &scene(&[8], IonicPolicy::CoresOnly, 33, 0.2)), minus_4, "O alone: the same surface");
        let mut c = private.lock().unwrap();
        assert_eq!(surface_hash(&mut c, IonicPolicy::CoresOnly), RECIPE_HASH_M7_4);
        let all = surface_hash(&mut c, IonicPolicy::All);
        assert_ne!(all, RECIPE_HASH_M7_4, "an All surface may hold the dimer-gated H: its provenance moves");
        assert_eq!(all, qm::radial::ledger::recipe_hash());
        assert!(!c.has_table(1), "no H was generated for the tag");
        drop(c);
        assert_eq!(tag_in(&private, &scene(&[1, 1], IonicPolicy::CoresOnly, 33, 0.25)), "none", "CoresOnly H₂ is the all-electron surface");
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
