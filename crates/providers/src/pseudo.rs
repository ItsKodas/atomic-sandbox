//! The process-wide pseudopotential cache (M7 Task 9; spec §5.5, `sim` bullet): tables are generated on first use
//! by `qm::radial::generate_with` on the calling thread (the worker or a side solver; seconds per element, plus the
//! dimer gate for H and He) and cached by (Z, recipe_hash); filtered tables by (Z, recipe_hash, h). Refusals are cached
//! too. Nothing is shipped (spec §3, rule 1): every table is made from this program's own all-electron atom. The one
//! thing kept on disk (when the app sets a directory, [`set_disk_cache`]) is the dimer gate's *derived* verdicts per
//! species identity — module doc, "The disk cache".
//!
//! # What is decided here
//! - [`setup_for`]: a scene's [`qm::IonicSetup`] under its [`sim::IonicPolicy`] — `None` when nothing is pseudised
//!   (policy `None`; `CoresOnly` with only core-less or refused species), so those scenes run today's all-electron
//!   path bit for bit. An `IonicError` (refused under `All`, h coarser than a table's `h_req`, a failed filtered
//!   gate) is a `ProviderError::Invalid`, raised before any solve.
//! - [`provenance_tag`]: the *surface* a scene is solved on, for the surrogate's provenance and the references' key.
//!   The tag follows the physics, not the menu: H₂ under `CoresOnly` has nothing pseudised and carries `"none"`.
//! - [`gpu_tol_floor`] / [`effective_gpu_tol`]: the floor-aware GPU stop (spec §7 risk 6).
//! - `Accepted` (M7 Task 11 fix round, the app's default): every species the recipe accepts, H and He included; a refused
//!   one runs all-electron with a note (`qm::ionic`). In a run without a GPU dimer backend H and He are not generated at
//!   all (≈ 40 min each on the CPU) but kept all-electron with the note `Refusal::DimerGateUnavailable` — the surface is
//!   then `CoresOnly`'s and the tag says so ("the tag follows the physics"); with the GPU it is `All`'s on every scene `All`
//!   can solve, and the tag is `All`'s (one surface identity per recipe and backend class, never per scene).
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
//! - **Procedure constants** (σ ladder, box, h, FD step, bisection width, tolerances, the N/A wall bound) are
//!   `qm::radial::ledger::DIMER_PROCEDURE`, hashed into `recipe_hash` (Task 11 review SHOULD-2).
//! - **Backend margin (Task 11 review Minor-4).** A verdict within ≈ 2e-3 Bohr in R_e or ≈ 0.5 % in ω of its bound is
//!   backend-dependent: the GPU's AE σ → 0 R_e is 1.4489 against the CPU oracle's 1.4511 (the ×88 extrapolation of a
//!   3e-6 per-solve force difference). Today's nearest verdict is H f = 1.4 polynomial, +0.0237 Bohr (+0.0216 against the
//!   CPU R_e): 0.0037 Bohr from the bound, a fail either way.
//! - **Not applicable.** Only when the AE dimer has no force zero in the bracket *and* the pseudised wall agrees at the
//!   outer end within `wall_rel` (Task 11 review Minor-2; He: 0.3 %); the end-sign test presumes the AE curve is monotone
//!   there (Z ≤ 2).
//!
//! # The disk cache (M7 Task 11 fix round; controller's cost condition)
//! The gate is the expensive, deterministic part of generating H and He (≈ 1 min on the GPU; the radial ladder itself takes
//! seconds). Its answers are *derived* data — a function of Z, the ledger and `DIMER_PROCEDURE`, i.e. of
//! `qm::radial::ledger::species_recipe_hash(z)` — not per-element input, so caching them is rule-1 clean (saved models are the
//! precedent). With a directory set ([`set_disk_cache`]; the app: its cache dir), `PseudoCache::table` for Z ≤
//! `dimer_z_max` reads `gate-z<Z>-<species_recipe_hash:016x>.txt` and replays the recorded answers into the ladder
//! ([`ReplayGate`]), regenerating the table bit for bit in seconds; an entry the file does not answer runs the real gate. A
//! file that does not parse, or names another Z or hash, is ignored and rewritten; answers that `Failed` are never written.
//! The validate suites and the tests set no directory.
use qm::ionic::{IonicPolicy as QmPolicy, IonicSetup};
use qm::radial::filter::{filtered, FilteredTable};
use qm::radial::{generate_with, has_core, ledger::recipe_hash as compute_recipe_hash, DimerError, DimerGate, DimerPair, GenerationRecord, LocalCandidate, PseudoTable, Refusal};
use qm::radial::ledger::{DIMER_PROCEDURE, LEDGER};
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

    /// The table of Z, generated on first use (a note for the next `setup_for` says so). For Z ≤ `dimer_z_max` through the
    /// dimer gate — replayed from the disk cache when it holds this species identity (module doc). While it runs,
    /// [`generating`] says so (the app's status line).
    pub(crate) fn table(&mut self, z: u8) -> Result<Arc<PseudoTable>, Refusal> {
        let key = (z, recipe_hash());
        if let Some(t) = self.tables.get(&key) {
            return t.clone();
        }
        let t0 = std::time::Instant::now();
        let gated = z <= qm::radial::ledger::LEDGER.dimer_z_max;
        let _status = Generating::start(z, gated);
        // the gate runs only for Z ≤ LEDGER.dimer_z_max: the ladder decides that, not the cache
        let mut gate = ReplayGate::new(z, GridDimerGate::new(dimer_backend()), if gated { disk_cache() } else { None });
        let t = generate_with(z, Some(&mut gate)).map(Arc::new);
        gate.store();
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
                if gate.inner.solves > 0 { format!("; dimer gate: {} ({} solves)", t.record.dimer, gate.inner.solves) }
                else if gate.replayed > 0 { format!("; dimer gate (from the disk cache): {}", t.record.dimer) } else { String::new() }
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
        wants_table(policy, z, &core) && self.table(z).is_ok()
    }
}

/// The one rule of which species `IonicSetup::build` hands a table to (spec §4.7): `All` every species, `Accepted` every
/// species except the dimer-gated ones in a run without a GPU gate (module doc; a recipe refusal is then the all-electron
/// fallback), `CoresOnly` those with a core (`core` = `has_core(z)`; a recipe refusal of the core test keeps the species
/// all-electron), `None` none. The tag, the memory guard and the chrome all go through it, so they cannot drift.
fn wants_table(policy: IonicPolicy, z: u8, core: &Result<bool, Refusal>) -> bool {
    wants_table_gpu(policy, z, core, dimer_backend_is_gpu())
}

/// [`wants_table`] with the gate's availability given (`gpu`: a GPU dimer backend is set).
fn wants_table_gpu(policy: IonicPolicy, z: u8, core: &Result<bool, Refusal>, gpu: bool) -> bool {
    match policy {
        IonicPolicy::None => false,
        IonicPolicy::All => true,
        IonicPolicy::Accepted => z > qm::radial::ledger::LEDGER.dimer_z_max || gpu,
        IonicPolicy::CoresOnly => matches!(core, Ok(true)),
    }
}

/// `sim`'s policy as `qm`'s.
pub fn qm_policy(p: IonicPolicy) -> QmPolicy {
    match p {
        IonicPolicy::None => QmPolicy::None,
        IonicPolicy::CoresOnly => QmPolicy::CoresOnly,
        IonicPolicy::All => QmPolicy::All,
        IonicPolicy::Accepted => QmPolicy::Accepted,
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
    let policy = scene.ionic;
    let built = IonicSetup::build(qm_policy(policy), scene.cell.h, sigma_ae, &zs, &mut |z| cores[&z].clone(), &mut |z| {
        // Accepted without a GPU gate: H and He are not generated (≈ 40 min each on the CPU) but kept all-electron (module doc)
        if !wants_table(policy, z, &Ok(true)) { return Err(Refusal::DimerGateUnavailable { z }); }
        c.filtered(z, scene.cell.h)
    });
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
        tag_with(&mut c, scene.ionic)
    } else {
        "none".into()
    }
}

/// The tag of a scene under `policy` with at least one species pseudised: the formatting half of
/// [`provenance_tag`], needing no scene (no table is generated) — what the viewer compares a model file's
/// ionic tag with before loading it. `"none"` for the policy `None`.
pub fn tag_of(policy: IonicPolicy) -> String {
    if policy == IonicPolicy::None {
        return "none".into();
    }
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    tag_with(&mut c, policy)
}

/// The tag's text for `policy` on a given cache (the surface word, the table-identity hash and the stop rules).
fn tag_with(c: &mut PseudoCache, policy: IonicPolicy) -> String {
    format!(
        "{}/{:016x}/shared{:e}/{:e}/stop2x{:e}",
        surface_word(c, policy),
        surface_hash(c, policy),
        qm::DEGENERACY_TAU,
        qm::scf::CANDIDATE_TAU,
        GPU_DRHO_FLOOR_PER_ELECTRON
    )
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
    if gates_species(c, policy) { recipe_hash() } else { qm::radial::ledger::core_recipe_hash() }
}

/// Whether `policy` can pseudise a dimer-gated species (Z ≤ `dimer_z_max`) in this run.
fn gates_species(c: &mut PseudoCache, policy: IonicPolicy) -> bool {
    (1..=qm::radial::ledger::LEDGER.dimer_z_max).any(|z| {
        let core = if policy == IonicPolicy::CoresOnly { c.has_core(z) } else { Ok(false) };
        wants_table(policy, z, &core)
    })
}

/// The tag's surface word: the policy's, except `Accepted`, whose surface is `All`'s (it can pseudise H/He: on every scene
/// `All` solves the two are the same physics) or, in a run without a GPU gate, `CoresOnly`'s (H and He all-electron) — the
/// tag follows the physics, so models are shared across the menus that solve the same surface.
fn surface_word(c: &mut PseudoCache, policy: IonicPolicy) -> &'static str {
    match policy {
        IonicPolicy::Accepted if gates_species(c, policy) => IonicPolicy::All.word(),
        IonicPolicy::Accepted => IonicPolicy::CoresOnly.word(),
        p => p.word(),
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
        if !wants_table(scene.ionic, z, &core) {
            n_valence += z as i64;
            continue;
        }
        match c.tables.get(&(z, hash)) {
            Some(Ok(t)) => {
                any = true;
                n_valence += t.z_v.round() as i64;
                ions.push(ion_extent(&c, z, h));
            }
            // refused: all-electron under CoresOnly and Accepted (an error under All, which the provider reports itself)
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

/// Whether a GPU backend is set: `Accepted` pseudises H and He only then (module doc).
pub fn dimer_backend_is_gpu() -> bool {
    matches!(dimer_backend(), DimerBackend::Gpu(_))
}

// ---- what is being generated now (the app's status line)

static GENERATING: Mutex<Option<(u8, std::time::Instant, bool)>> = Mutex::new(None);

/// The species whose table is being generated right now, the seconds so far, and whether its dimer gate runs (≈ 1 min on
/// the GPU — the app's status line says so while it lasts). `None` when nothing is generated.
pub fn generating() -> Option<(u8, f64, bool)> {
    GENERATING.lock().unwrap_or_else(|e| e.into_inner()).map(|(z, t, g)| (z, t.elapsed().as_secs_f64(), g))
}

/// Sets [`generating`] for its lifetime (cleared on drop, a panic included).
struct Generating;

impl Generating {
    fn start(z: u8, gated: bool) -> Generating {
        *GENERATING.lock().unwrap_or_else(|e| e.into_inner()) = Some((z, std::time::Instant::now(), gated));
        Generating
    }
}

impl Drop for Generating {
    fn drop(&mut self) {
        *GENERATING.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

// ---- the disk cache of the gate's derived answers (module doc, "The disk cache")

static DISK: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

/// The directory of the dimer gate's disk cache (module doc), or `None` (the default: no disk cache).
pub fn set_disk_cache(dir: Option<std::path::PathBuf>) {
    *DISK.lock().unwrap_or_else(|e| e.into_inner()) = dir;
}

/// The disk cache's directory, if set.
pub fn disk_cache() -> Option<std::path::PathBuf> {
    DISK.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The app's cache directory for derived pseudopotential data: `ATOMIC_SANDBOX_CACHE_DIR` if set, else
/// `%LOCALAPPDATA%\atomic-sandbox\cache\pseudo` (Windows), `$XDG_CACHE_HOME/atomic-sandbox/pseudo` or
/// `$HOME/.cache/atomic-sandbox/pseudo`; `None` when none of these exists.
pub fn default_disk_cache_dir() -> Option<std::path::PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
    var("ATOMIC_SANDBOX_CACHE_DIR")
        .or_else(|| var("LOCALAPPDATA").map(|p| p.join("atomic-sandbox").join("cache").join("pseudo")))
        .or_else(|| var("XDG_CACHE_HOME").map(|p| p.join("atomic-sandbox").join("pseudo")))
        .or_else(|| var("HOME").map(|p| p.join(".cache").join("atomic-sandbox").join("pseudo")))
}

/// One recorded answer: the ladder entry (f, local) and what the gate said.
type Answer = (f64, LocalCandidate, Result<DimerPair, DimerError>);

const DISK_MAGIC: &str = "atomic-sandbox dimer gate v1";

/// The cache file of Z at this build's species identity.
fn disk_file(dir: &std::path::Path, z: u8) -> std::path::PathBuf {
    dir.join(format!("gate-z{z}-{:016x}.txt", qm::radial::ledger::species_recipe_hash(z)))
}

fn one_line(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// The file's text for `answers` (module doc, "The disk cache"). f64 in `Debug` (shortest round trip).
fn write_answers(z: u8, answers: &[Answer]) -> String {
    let mut s = format!("{DISK_MAGIC}\nz\t{z}\nspecies_recipe_hash\t{:016x}\nprocedure\t{}\n", qm::radial::ledger::species_recipe_hash(z), one_line(&format!("{DIMER_PROCEDURE:?}")));
    for (f, local, a) in answers {
        let tail = match a {
            Ok(p) => format!("pair\t{:?}\t{:?}\t{:?}\t{:?}", p.pp.0, p.pp.1, p.ae.0, p.ae.1),
            Err(DimerError::NotApplicable(m)) => format!("not-applicable\t{}", one_line(m)),
            Err(DimerError::PseudoUnbound(m)) => format!("pseudo-unbound\t{}", one_line(m)),
            Err(DimerError::WallMismatch(m)) => format!("wall-mismatch\t{}", one_line(m)),
            Err(DimerError::Failed(m)) => format!("failed\t{}", one_line(m)),
        };
        s += &format!("entry\t{f:?}\t{local:?}\t{tail}\n");
    }
    s
}

/// The answers of a cache file for Z at this build's identity; `None` for anything that does not parse, another Z or
/// another species identity (the caller then runs the real gate and rewrites the file). `Failed` answers are refused.
fn read_answers(text: &str, z: u8) -> Option<Vec<Answer>> {
    let mut lines = text.lines();
    if lines.next()? != DISK_MAGIC { return None; }
    let field = |l: Option<&str>, k: &str| -> Option<String> { let (a, b) = l?.split_once('\t')?; (a == k).then(|| b.to_string()) };
    if field(lines.next(), "z")?.parse::<u8>().ok()? != z { return None; }
    if u64::from_str_radix(&field(lines.next(), "species_recipe_hash")?, 16).ok()? != qm::radial::ledger::species_recipe_hash(z) { return None; }
    field(lines.next(), "procedure")?;
    let mut out = Vec::new();
    for l in lines {
        let c: Vec<&str> = l.split('\t').collect();
        if c.len() < 5 || c[0] != "entry" { return None; }
        let f: f64 = c[1].parse().ok()?;
        let local = match c[2] { "HighestChannel" => LocalCandidate::HighestChannel, "Polynomial" => LocalCandidate::Polynomial, _ => return None };
        let a = match c[3] {
            "pair" if c.len() == 8 => {
                let x: Vec<f64> = c[4..8].iter().map(|v| v.parse::<f64>()).collect::<Result<_, _>>().ok()?;
                if x.iter().any(|v| !v.is_finite()) { return None; }
                Ok(DimerPair { pp: (x[0], x[1]), ae: (x[2], x[3]) })
            }
            "not-applicable" => Err(DimerError::NotApplicable(c[4..].join(" "))),
            "pseudo-unbound" => Err(DimerError::PseudoUnbound(c[4..].join(" "))),
            "wall-mismatch" => Err(DimerError::WallMismatch(c[4..].join(" "))),
            _ => return None,
        };
        out.push((f, local, a));
    }
    Some(out)
}

/// The gate the cache generates with (module doc, "The disk cache"): answers recorded on disk for this species identity are
/// replayed, anything else goes to the real [`GridDimerGate`]; [`ReplayGate::store`] writes what was asked when the real
/// gate had to run.
pub struct ReplayGate {
    z: u8,
    pub inner: GridDimerGate,
    file: Option<std::path::PathBuf>,
    recorded: Vec<Answer>,
    asked: Vec<Answer>,
    /// Answers taken from the file.
    pub replayed: usize,
    missed: bool,
}

impl ReplayGate {
    /// The replaying gate of Z over `inner`, reading `dir`'s file when a directory is given.
    pub fn new(z: u8, inner: GridDimerGate, dir: Option<std::path::PathBuf>) -> ReplayGate {
        let file = dir.map(|d| disk_file(&d, z));
        let recorded = file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()).and_then(|t| read_answers(&t, z)).unwrap_or_default();
        ReplayGate { z, inner, file, recorded, asked: Vec::new(), replayed: 0, missed: false }
    }

    /// Writes the asked answers when the real gate ran and none failed (a failure is never cached); best effort — an I/O
    /// error only means the next run pays the gate again.
    pub fn store(&self) {
        let Some(file) = &self.file else { return };
        if !self.missed || self.asked.is_empty() || self.asked.iter().any(|a| matches!(a.2, Err(DimerError::Failed(_)))) { return; }
        let tmp = file.with_extension("tmp");
        let ok = file.parent().is_some_and(|d| std::fs::create_dir_all(d).is_ok())
            && std::fs::write(&tmp, write_answers(self.z, &self.asked)).is_ok()
            && std::fs::rename(&tmp, file).is_ok();
        if !ok { eprintln!("pseudopotential disk cache: could not write {}", file.display()); }
    }
}

impl DimerGate for ReplayGate {
    fn dimers(&mut self, t: &PseudoTable) -> Result<DimerPair, DimerError> {
        if let Some((_, _, a)) = self.recorded.iter().find(|(f, l, _)| *f == t.f && *l == t.local) {
            self.replayed += 1;
            self.asked.push((t.f, t.local, a.clone()));
            return a.clone();
        }
        self.missed = true;
        let a = self.inner.dimers(t);
        self.asked.push((t.f, t.local, a.clone()));
        a
    }
}

// ---- the 3-D computation

/// The gate's grid (module doc; `DIMER_PROCEDURE`): `LEDGER.h_ladder[h_ladder_index]` on a `box_bohr` cube.
fn finest_grid() -> (qm::Grid, f64) {
    let h = LEDGER.h_ladder[DIMER_PROCEDURE.h_ladder_index];
    (qm::Grid::cube(2 * (DIMER_PROCEDURE.box_bohr / (2.0 * h)).round() as usize + 1, h), h)
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

    fn settings(&self) -> qm::ScfSettings {
        let mut s = qm::ScfSettings::for_grid(&self.grid);
        s.tol = DIMER_PROCEDURE.scf_tol;
        s.eig_tol = DIMER_PROCEDURE.eig_tol;
        s
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
        let mut s = self.settings();
        s.occupations = qm::Occupations::DegenerateShells { tau: qm::DEGENERACY_TAU };
        self.at(&qm::System::with_ionic(dimer_nuclei(z, r), 0, None, setup.clone()), &s)
    }

    /// The all-electron dimer's bond force at R, extrapolated σ → 0 (module doc, "All-electron dimer").
    fn all_electron(&mut self, z: u8, r: f64) -> Result<f64, String> {
        let h = self.grid.h;
        let sigmas = DIMER_PROCEDURE.sigma_ladder_over_h.map(|f| f * h);
        let sys = qm::System::new(dimer_nuclei(z, r), 0, None);
        let mut fs = [0.0; 4];
        for (k, &sigma) in sigmas.iter().enumerate() {
            let mut s = self.settings();
            s.sigma = sigma;
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
    let b = LEDGER.dimer_bracket;
    let r_peak = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.r_peak).fold(0.0, f64::max);
    (b[0] * r_peak, b[1] * r_peak)
}

/// Where the force is zero, or why there is no zero in the bracket.
#[derive(Clone, Debug, PartialEq)]
enum Located { Bracketed(f64), Unbracketed { lo: f64, hi: f64, f_lo: f64, f_hi: f64 } }

type Force<'a> = &'a mut dyn FnMut(f64) -> Result<f64, String>;

fn logged(force: Force, r: f64, log: &mut Vec<String>) -> Result<f64, String> {
    let v = force(r).map_err(|e| format!("R = {r:.4} Bohr: {e}"))?;
    log.push(format!("R {r:.5} Bohr: F {v:+.6e} Ha/Bohr"));
    Ok(v)
}

/// R_e by bisection of the sign of F in [lo, hi] to `bisection_over_dimer_re`·`dimer_re` and linear interpolation in the
/// last bracket (module doc); `Unbracketed` unless F(lo) > 0 > F(hi).
fn locate_re(lo: f64, hi: f64, log: &mut Vec<String>, force: Force) -> Result<Located, String> {
    let (mut a, mut b) = (lo, hi);
    let (mut fa, mut fb) = (logged(force, a, log)?, logged(force, b, log)?);
    if !(fa > 0.0 && fb < 0.0) {
        return Ok(Located::Unbracketed { lo, hi, f_lo: fa, f_hi: fb });
    }
    while b - a > DIMER_PROCEDURE.bisection_over_dimer_re * LEDGER.dimer_re {
        let m = 0.5 * (a + b);
        let fm = logged(force, m, log)?;
        if fm > 0.0 { (a, fa) = (m, fm) } else { (b, fb) = (m, fm) }
    }
    Ok(Located::Bracketed(a + (b - a) * fa / (fa - fb)))
}

/// k = −[F(R_e + δ) − F(R_e − δ)]/(2δ) (Ha/Bohr²).
fn k_at(r_e: f64, delta: f64, log: &mut Vec<String>, force: Force) -> Result<f64, String> {
    Ok(-(logged(force, r_e + delta, log)? - logged(force, r_e - delta, log)?) / (2.0 * delta))
}

/// μ = m/2 of the homonuclear dimer (a.u.), m the most abundant isotope's mass.
fn reduced_mass(z: u8) -> Result<f64, String> {
    Ok(0.5 * units::amu_to_au_mass(units::isotope_mass_amu(z).ok_or_else(|| format!("no isotope mass for Z = {z}"))?))
}

fn omega(k: f64, mu: f64, r_e: f64) -> Result<f64, String> {
    if !(k > 0.0) { return Err(format!("k = {k:.3e} Ha/Bohr² ≤ 0 at R_e {r_e:.4} Bohr")); }
    Ok((k / mu).sqrt())
}

/// A dimer's minimum and its frequency three ways (Task 11 review SHOULD-1): ω at the gate's step δ = `fd_step_over_h`·h,
/// at δ/2, and their Richardson extrapolation k₀ = [4k(δ/2) − k(δ)]/3, which removes the O(δ²) bias of the central
/// difference without a curve model (k(δ) = k + E⁗δ²/6 + O(δ⁴)). Atomic units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimerFit { pub r_e: f64, pub omega: f64, pub omega_half: f64, pub omega_richardson: f64 }

/// The gate's (R_e, ω) — or `Unbracketed` — of the dimer with bond force `force`; with `richardson` also the step-halved
/// and extrapolated frequencies (2 more R points).
fn fit(z: u8, lo: f64, hi: f64, h: f64, richardson: bool, log: &mut Vec<String>, force: Force) -> Result<Result<DimerFit, Located>, String> {
    let r_e = match locate_re(lo, hi, log, force)? {
        Located::Bracketed(r) => r,
        u => return Ok(Err(u)),
    };
    let (mu, delta) = (reduced_mass(z)?, DIMER_PROCEDURE.fd_step_over_h * h);
    let k = k_at(r_e, delta, log, force)?;
    let w = omega(k, mu, r_e)?;
    if !richardson { return Ok(Ok(DimerFit { r_e, omega: w, omega_half: f64::NAN, omega_richardson: f64::NAN })); }
    let k2 = k_at(r_e, 0.5 * delta, log, force)?;
    Ok(Ok(DimerFit { r_e, omega: w, omega_half: omega(k2, mu, r_e)?, omega_richardson: omega((4.0 * k2 - k) / 3.0, mu, r_e)? }))
}

fn unbracketed(u: Located) -> String {
    match u {
        Located::Unbracketed { lo, hi, f_lo, f_hi } => format!("no force zero in [{lo:.3}, {hi:.3}] Bohr: F({lo:.3}) {f_lo:+.3e}, F({hi:.3}) {f_hi:+.3e} Ha/Bohr"),
        Located::Bracketed(r) => format!("bracketed at {r:.4}"),
    }
}

/// The pseudised homonuclear dimer of `t` by the gate's procedure (module doc): (R_e Bohr, ω a.u.).
pub fn pseudo_dimer(t: &PseudoTable, backend: &DimerBackend) -> Result<(f64, f64), String> {
    pseudo_dimer_fit_with(t, backend, false).map(|f| (f.r_e, f.omega))
}

/// [`pseudo_dimer`] with the step-halved and Richardson frequencies ([`DimerFit`]).
pub fn pseudo_dimer_fit(t: &PseudoTable, backend: &DimerBackend) -> Result<DimerFit, String> {
    pseudo_dimer_fit_with(t, backend, true)
}

fn pseudo_dimer_fit_with(t: &PseudoTable, backend: &DimerBackend, richardson: bool) -> Result<DimerFit, String> {
    let mut solver = BondForce::new(backend.clone());
    let h = solver.grid.h;
    let setup = pseudo_setup(t, h);
    let (lo, hi) = bracket(t);
    fit(t.z, lo, hi, h, richardson, &mut Vec::new(), &mut |r| solver.pseudo(t.z, &setup, r))?.map_err(unbracketed)
}

/// The all-electron homonuclear dimer of Z (= `t.z`; `t` gives the bracket from its AE reference) at σ → 0 by the gate's
/// procedure, with the step-halved and Richardson frequencies ([`DimerFit`]).
pub fn all_electron_dimer_fit(t: &PseudoTable, backend: &DimerBackend) -> Result<DimerFit, String> {
    let mut solver = BondForce::new(backend.clone());
    let h = solver.grid.h;
    let (lo, hi) = bracket(t);
    fit(t.z, lo, hi, h, true, &mut Vec::new(), &mut |r| solver.all_electron(t.z, r))?.map_err(unbracketed)
}

/// What the gate knows of an element's all-electron dimer (computed once per Z per gate).
#[derive(Clone, Debug)]
enum Ae { Bound((f64, f64)), Unbracketed(Located), Failed(String) }

/// The grid dimer gate (module doc): `qm::radial::DimerGate` for `generate_with`. The all-electron dimer is computed
/// once per Z for the gate's lifetime (it does not depend on f). `log` holds every bond-force evaluation and verdict,
/// `solves` counts the SCF solves (reported).
pub struct GridDimerGate {
    pub backend: DimerBackend,
    ae_cache: HashMap<u8, Ae>,
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
            log.push(format!("{sym}₂ all-electron σ → 0 at h = {h} (σ = h·{:?}), bracket [{lo:.4}, {hi:.4}] Bohr:", DIMER_PROCEDURE.sigma_ladder_over_h));
            let mut trace = Vec::new();
            let out = fit(t.z, lo, hi, h, false, log, &mut |r| {
                let f0 = solver.all_electron(t.z, r);
                trace.push(format!("  R {r:.5}: {} → F₀ {}", solver.last_sigma, f0.as_ref().map_or_else(|e| e.clone(), |f| format!("{f:+.6e}"))));
                f0
            });
            log.extend(trace);
            match out {
                Ok(Ok(f)) => Ae::Bound((f.r_e, f.omega)),
                Ok(Err(u)) => Ae::Unbracketed(u),
                Err(e) => Ae::Failed(e),
            }
        }).clone();
        log.push(format!("{sym}₂ pseudised, f = {}, {:?}:", t.f, t.local));
        let setup = pseudo_setup(t, h);
        let out = match ae {
            Ae::Bound(ae) => match fit(t.z, lo, hi, h, false, log, &mut |r| solver.pseudo(t.z, &setup, r)) {
                Ok(Ok(pp)) => Ok(DimerPair { pp: (pp.r_e, pp.omega), ae }),
                Ok(Err(u)) => Err(DimerError::PseudoUnbound(unbracketed(u))),
                Err(e) => Err(DimerError::Failed(e)),
            },
            // not applicable — if the pseudised repulsive wall agrees with the AE one at the outer end (review Minor-2); the
            // PP F at both ends recorded (plan review Q9(ii))
            Ae::Unbracketed(u) => {
                let Located::Unbracketed { f_hi, .. } = u else { unreachable!("an unbracketed AE dimer") };
                match (solver.pseudo(t.z, &setup, lo), solver.pseudo(t.z, &setup, hi)) {
                    (Ok(p_lo), Ok(p_hi)) => {
                        let wall = p_hi / f_hi - 1.0;
                        let why = format!("AE F (σ → 0): {}; PP F({lo:.3}) {p_lo:+.3e}, PP F({hi:.3}) {p_hi:+.3e} Ha/Bohr; repulsive wall at the outer end PP/AE − 1 = {:+.2} % (bound ±{} %)",
                                          unbracketed(u.clone()), 100.0 * wall, 100.0 * DIMER_PROCEDURE.wall_rel);
                        if wall.abs() <= DIMER_PROCEDURE.wall_rel { Err(DimerError::NotApplicable(why)) } else { Err(DimerError::WallMismatch(why)) }
                    }
                    (Err(e), _) | (_, Err(e)) => Err(DimerError::Failed(e)),
                }
            }
            Ae::Failed(e) => Err(DimerError::Failed(e)),
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
            wants_table(scene.ionic, z, &core).then(|| summary_of(z, t.z_v, t.record.r_core, &t.record))
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
        // the dimer gate (Task 11) moved only H and He, through `DIMER_PROCEDURE`, not the version: the O, N, C figures stand
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
        // review Minor-2: "not applicable" only with the pseudised repulsive wall agreeing at the outer bracket end
        assert!(he.record.dimer.contains("repulsive wall at the outer end"), "{}", he.record.dimer);
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

    /// Task 11 fix round (controller's default ruling): `Accepted` pseudises every species the recipe accepts — H and He only
    /// when the run has a GPU dimer gate (else all-electron with a note, never a 40-min CPU gate) — and its surface is `All`'s
    /// (tag word "all", the full recipe hash) or, without the gate, `CoresOnly`'s ("cores", the core hash).
    #[test]
    fn accepted_pseudises_what_the_recipe_accepts_and_names_its_surface() {
        let ok = Ok(true);
        for z in [1u8, 2] {
            assert!(wants_table_gpu(IonicPolicy::Accepted, z, &Ok(false), true), "Z = {z} with the GPU gate");
            assert!(!wants_table_gpu(IonicPolicy::Accepted, z, &Ok(false), false), "Z = {z} without it: all-electron");
            assert!(wants_table_gpu(IonicPolicy::All, z, &Ok(false), false), "All generates on the CPU when asked to");
        }
        for z in [3u8, 6, 8, 13] {
            assert!(wants_table_gpu(IonicPolicy::Accepted, z, &ok, false) && wants_table_gpu(IonicPolicy::Accepted, z, &ok, true), "Z = {z}");
        }
        assert!(!wants_table_gpu(IonicPolicy::CoresOnly, 1, &Ok(false), true) && !wants_table_gpu(IonicPolicy::None, 8, &ok, true));
        assert_eq!(qm_policy(IonicPolicy::Accepted), QmPolicy::Accepted);
    }

    /// Task 11 fix round (controller's cost condition): the disk cache's file round-trips every kind of answer, and anything
    /// that does not parse, names another Z or another species identity is ignored (the gate then runs and rewrites it).
    #[test]
    fn the_gate_disk_cache_round_trips_and_rejects_what_does_not_match() {
        let answers: Vec<Answer> = vec![
            (1.4, LocalCandidate::HighestChannel, Ok(DimerPair { pp: (1.3931183157135343, 0.019237754435840068), ae: (1.4489293112594686, 0.019401790957326262) })),
            (1.4, LocalCandidate::Polynomial, Err(DimerError::PseudoUnbound("no force zero	in [1, 4]".into()))),
            (1.2, LocalCandidate::HighestChannel, Err(DimerError::NotApplicable("AE F (σ → 0): no zero; PP F ...".into()))),
            (1.0, LocalCandidate::Polynomial, Err(DimerError::WallMismatch("wall +9 %".into()))),
        ];
        let text = write_answers(1, &answers);
        let back = read_answers(&text, 1).expect("round trip");
        assert_eq!(back.len(), 4);
        assert_eq!(back[0], answers[0], "f64 in Debug round-trips exactly");
        assert_eq!(back[1].2, Err(DimerError::PseudoUnbound("no force zero in [1, 4]".into())), "tabs are flattened");
        assert_eq!(back[2], answers[2]);
        assert_eq!(back[3], answers[3]);
        assert!(read_answers(&text, 2).is_none(), "another Z");
        assert!(read_answers(&text.replacen(&format!("{:016x}", qm::radial::ledger::species_recipe_hash(1)), "0000000000000000", 1), 1).is_none(), "another identity");
        assert!(read_answers("garbage", 1).is_none() && read_answers("", 1).is_none());
        assert!(read_answers(&format!("{text}entry	1.2	HighestChannel	pair	NaN	1	1	1
"), 1).is_none(), "a non-finite pair");
        assert!(read_answers(&format!("{text}entry	1.2	HighestChannel	failed	device lost
"), 1).is_none(), "a failure is never cached");
    }

    /// Task 11 fix round: a cached answer is replayed into the ladder — the table regenerates with the recorded verdict and the
    /// real gate never runs (here a CPU gate that would take ≈ 40 min: a single solve would show in `inner.solves`); the file
    /// is not rewritten when nothing was missed. A file for another identity is ignored.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (the H ladder); cargo test -p providers --release")]
    fn a_cached_gate_answer_is_replayed_without_solving() {
        let ungated = qm::radial::generate(1).expect("H passes the atomic gates");
        let dir = std::env::temp_dir().join(format!("atomic-sandbox-gate-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pair = DimerPair { pp: (1.44, 0.0190), ae: (1.45, 0.0194) };
        let file = disk_file(&dir, 1);
        std::fs::write(&file, write_answers(1, &[(ungated.f, ungated.local, Ok(pair))])).unwrap();
        let before = std::fs::metadata(&file).unwrap().modified().unwrap();
        let mut gate = ReplayGate::new(1, GridDimerGate::new(DimerBackend::Cpu), Some(dir.clone()));
        let t = qm::radial::generate_with(1, Some(&mut gate)).expect("H generates from the cache");
        gate.store();
        assert_eq!((gate.replayed, gate.inner.solves), (1, 0));
        assert_eq!((t.f, t.local, t.provisional), (ungated.f, ungated.local, false));
        assert!(t.record.dimer.starts_with("pass: PP R_e 1.4400"), "{}", t.record.dimer);
        assert_eq!(std::fs::metadata(&file).unwrap().modified().unwrap(), before, "nothing missed: not rewritten");
        // the same table as a run that computed this verdict (the radial ladder is deterministic)
        assert_eq!(t.kb.v_loc, ungated.kb.v_loc);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Task 11 fix round (the controller's condition on the default): the gated H is generated at most once per cache —
    /// i.e. once per process for the shared one, whatever number of tests or scenes ask — so the app and worker suites under
    /// the `Accepted` default pay the ≈ 1 min gate once per binary. Two setups and a tag of an `Accepted` H₂ scene on one
    /// (private) cache: one generation, the generation note once, `generating()` clear afterwards. GPU (skips without one).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (the H dimer gate on the GPU); cargo test -p providers --release")]
    fn the_gated_h_is_generated_once_per_cache() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        set_dimer_backend(DimerBackend::Gpu(ctx));
        let private = Mutex::new(PseudoCache::default());
        let h2 = scene(&[1, 1], IonicPolicy::Accepted, 61, 0.2);
        let t0 = std::time::Instant::now();
        let (mut first, mut second) = (Vec::new(), Vec::new());
        let a = setup_in(&private, &h2, 0.2, &mut first).expect("H₂ under Accepted").expect("H pseudised");
        let t_first = t0.elapsed().as_secs_f64();
        let b = setup_in(&private, &h2, 0.2, &mut second).expect("again").expect("again");
        let tag = tag_in(&private, &h2);
        eprintln!("first setup {t_first:.0} s ({first:?}); second {:.2} s; tag {tag}", t0.elapsed().as_secs_f64() - t_first);
        assert_eq!(private.lock().unwrap().generated(), 1, "the H table was generated once");
        assert!(first.iter().any(|n| n.starts_with("generated the H pseudopotential") && n.contains("dimer gate")), "{first:?}");
        assert!(!second.iter().any(|n| n.starts_with("generated")), "{second:?}");
        assert_eq!(a.pseudised(), b.pseudised());
        assert!(tag.starts_with("all/"), "Accepted with the GPU gate is All's surface: {tag}");
        assert!(generating().is_none(), "the status clears after generation");
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
