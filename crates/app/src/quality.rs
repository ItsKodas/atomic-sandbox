//! Quality presets (user request 2026-10-14, "Quality presets" in
//! `docs/superpowers/notes/2026-10-13-m5-user-requests.md`): the grid spacing h as a choice
//! between faster-but-coarser and slower-but-more-accurate, with the box's physical extent kept.
//!
//! - **Fast** h = 0.30, **Balanced** h = 0.25 (the default before 2026-10-14), **Accurate**
//!   h = 0.15 (**the default**), **Precise** h = 0.10, and **Custom** h ∈ [0.08, 0.35] Bohr,
//!   snapped to 3 decimals (so the surrogate's model provenance keys stay canonical).
//! - σ follows h (`Scene::sigma = None` means σ = h); an explicit σ is never overwritten.
//! - The box is kept: n is the smallest odd integer with (n − 1)·h ≥ the box length L, so a
//!   quality change never shrinks the cell (L = 12 Bohr for the presets: 41³, 49³, 81³, 121³).
//! - The accuracy the levels buy is measured, not guessed: H₂'s bond length on this solver at
//!   σ = h (`crates/validate/references.md`, "Quality presets") against 0.741 Å (experiment).
use crate::scene::Scene;

/// The GPU memory budget (GB) when neither `ATOMIC_SANDBOX_VRAM_GB` is set nor the adapter
/// reports its memory ([`set_adapter_memory`]): half the project's 24 GB RX 7900 XTX
/// (`providers::ks`, "GPU memory"). Set the variable on such a card.
pub const VRAM_BUDGET_GB: f64 = 12.0;

/// The share of the card's memory the budget takes when the adapter reports it (user request
/// 2026-10-19): half, as [`VRAM_BUDGET_GB`] is of the project's card — the rest is the
/// renderer's, the driver's and other applications'.
pub const ADAPTER_SHARE: f64 = 0.5;

/// The card's device-local memory (bytes; 0 = not reported), set once a device exists
/// (`Viewer::start`, from `GpuContext::device_local_memory_bytes`).
static ADAPTER_MEMORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records the card's memory for [`vram_budget`] (`None`: the backend did not report it).
pub fn set_adapter_memory(bytes: Option<u64>) {
    ADAPTER_MEMORY.store(bytes.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// Where the GPU memory budget came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BudgetSource {
    /// `ATOMIC_SANDBOX_VRAM_GB`.
    Env,
    /// [`ADAPTER_SHARE`] of the card's reported memory (bytes).
    Adapter { card_bytes: u64 },
    /// [`VRAM_BUDGET_GB`]: the card's memory is not known.
    Default,
}

/// The GPU memory budget and its source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VramBudget {
    pub bytes: u64,
    pub source: BudgetSource,
}

impl VramBudget {
    /// A budget of `bytes` with no stated source (the tests).
    #[cfg(test)]
    pub fn fixed(bytes: u64) -> VramBudget {
        VramBudget { bytes, source: BudgetSource::Default }
    }

    /// "12.9 GB" — the budget as the refusals print it.
    pub fn gb(&self) -> String {
        format!("{:.1} GB", self.bytes as f64 / 1e9)
    }

    /// Where it came from, for a refusal: "half of this GPU's 25.8 GB; ATOMIC_SANDBOX_VRAM_GB
    /// overrides it".
    pub fn why(&self) -> String {
        match self.source {
            BudgetSource::Env => "set by ATOMIC_SANDBOX_VRAM_GB".into(),
            BudgetSource::Adapter { card_bytes } => {
                format!("{:.0} % of this GPU's {:.1} GB; ATOMIC_SANDBOX_VRAM_GB overrides it", 100.0 * ADAPTER_SHARE, card_bytes as f64 / 1e9)
            }
            BudgetSource::Default => "the default: this GPU did not report its memory; set ATOMIC_SANDBOX_VRAM_GB".into(),
        }
    }
}

/// The GPU memory budget: `ATOMIC_SANDBOX_VRAM_GB` (a positive number of GB), else
/// [`ADAPTER_SHARE`] of the card's memory when the adapter reported it, else [`VRAM_BUDGET_GB`].
pub fn vram_budget() -> VramBudget {
    let env = std::env::var("ATOMIC_SANDBOX_VRAM_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|x| x.is_finite() && *x > 0.0);
    budget_from(env, ADAPTER_MEMORY.load(std::sync::atomic::Ordering::Relaxed))
}

/// [`vram_budget`] from its inputs: the variable's GB and the card's bytes (0 = unknown).
pub fn budget_from(env_gb: Option<f64>, card_bytes: u64) -> VramBudget {
    match env_gb {
        Some(gb) => VramBudget { bytes: (gb * 1e9) as u64, source: BudgetSource::Env },
        None if card_bytes > 0 => VramBudget { bytes: (card_bytes as f64 * ADAPTER_SHARE) as u64, source: BudgetSource::Adapter { card_bytes } },
        None => VramBudget { bytes: (VRAM_BUDGET_GB * 1e9) as u64, source: BudgetSource::Default },
    }
}

/// GPU solvers a backend runs: 1 for the GPU backend, 2 for the estimated solver with a GPU
/// teacher (the teacher and its background auditor, each a full `GpuKs`), 0 on the CPU.
pub fn gpu_solvers(backend: crate::sim::Backend, gpu_available: bool) -> u64 {
    match backend {
        crate::sim::Backend::Gpu => 1,
        crate::sim::Backend::Estimated if gpu_available => 2,
        _ => 0,
    }
}

/// `Err(message)` when `scene` on `solvers` GPU solvers would exceed `budget` (or has an element
/// without a mass, which the worker refuses anyway: no estimate then). The message names the
/// budget and where it came from.
pub fn vram_check(scene: &Scene, solvers: u64, budget: VramBudget) -> Result<(), String> {
    let Ok(s) = scene.to_sim() else { return Ok(()) };
    let need = solvers * providers::gpu_footprint_bytes(&s);
    let fp = providers::pseudo::footprint_inputs(&s);
    if solvers > 0 && need > budget.bytes {
        return Err(format!(
            "refused: {}³ × {:.3} Bohr needs ≈ {:.1} GB of GPU memory ({} solver{}, {} {}electrons), over the {} budget ({}) — choose a coarser quality or fewer atoms",
            scene.grid_n,
            scene.grid_h,
            need as f64 / 1e9,
            solvers,
            if solvers == 1 { "" } else { "s: teacher + auditor" },
            fp.n_valence.max(0),
            if fp.pseudised { "valence " } else { "" },
            budget.gb(),
            budget.why()
        ));
    }
    Ok(())
}

/// The named levels: (name, h in Bohr).
pub const LEVELS: [(&str, f64); 4] = [("fast", 0.30), ("balanced", 0.25), ("accurate", 0.15), ("precise", 0.10)];

/// The default grid spacing (Bohr): Accurate.
pub const DEFAULT_H: f64 = 0.15;

/// The Custom slider's range (Bohr).
pub const H_MIN: f64 = 0.08;
pub const H_MAX: f64 = 0.35;

/// h snapped to 3 decimals and clamped to [`H_MIN`, `H_MAX`] (a non-finite h becomes the default).
pub fn snap(h: f64) -> f64 {
    if !h.is_finite() {
        return DEFAULT_H;
    }
    ((h.clamp(H_MIN, H_MAX) * 1000.0).round() / 1000.0).clamp(H_MIN, H_MAX)
}

/// The level named `name` (case-insensitive): its h.
pub fn parse(name: &str) -> Option<f64> {
    let name = name.trim().to_ascii_lowercase();
    LEVELS.iter().find(|(n, _)| *n == name).map(|&(_, h)| h)
}

/// The level whose h is `h` (to 3 decimals), or `None` (Custom).
pub fn level_of(h: f64) -> Option<&'static str> {
    LEVELS.iter().find(|(_, x)| (x - h).abs() < 5e-4).map(|&(n, _)| n)
}

/// "Accurate" for "accurate".
pub fn title(name: &str) -> String {
    let mut c = name.chars();
    c.next().map_or(String::new(), |f| f.to_ascii_uppercase().to_string() + c.as_str())
}

/// The smallest odd n with (n − 1)·h ≥ `box_len` (1e-9 relative slack so 12 / 0.15 is 80, not
/// 81 from the rounding of 0.15).
pub fn grid_n(box_len: f64, h: f64) -> usize {
    let k = (box_len / h * (1.0 - 1e-9)).ceil().max(2.0) as usize;
    let k = if k % 2 == 1 { k + 1 } else { k };
    k + 1
}

/// The box length (n − 1)·h of a scene (Bohr).
pub fn box_len(s: &Scene) -> f64 {
    (s.grid_n as f64 - 1.0) * s.grid_h
}

/// `scene` at spacing h (snapped) with its box length `box_len` kept (rounded up to whole odd n);
/// σ untouched (None follows h; an explicit value stays).
pub fn regrid(scene: &Scene, box_len: f64, h: f64) -> Scene {
    let h = snap(h);
    Scene { grid_n: grid_n(box_len, h), grid_h: h, ..scene.clone() }
}

/// A preset at quality h (the app's presets; `Scene::preset` is the file as written).
pub fn preset(name: &str, h: f64) -> Option<Scene> {
    let s = Scene::preset(name)?;
    let l = box_len(&s);
    Some(regrid(&s, l, h))
}

/// The cost of a solve at n′³ estimated from a measured one at n³: × (n′/n)³ (the work per SCF
/// step is linear in the points; the step count is assumed unchanged). A rough guide: below ~10⁶
/// points the GPU is latency-bound, so scaling up from a coarse solve overestimates (49³ → 81³:
/// 0.68 s predicted, 0.53 measured; 81³ → 121³: 1.77 vs 0.76 — `references.md` "Quality presets").
pub fn scaled_cost(seconds: f64, n_from: usize, n_to: usize) -> f64 {
    seconds * (n_to as f64 / n_from as f64).powi(3)
}

// ---- The estimated solver's box (user decision 2026-11-14) ----
//
// A surrogate sample is box-affected — not trained on (`TeachOutcome::BoxNotKept`) — when the
// teacher's density on the outermost grid plane exceeds the box rule
// `providers::box_affected_face_density(h)` (`BOX_AFFECTED_FACE_DENSITY` = 2e-5 at h = 0.25,
// scaled by (h / 0.25)²: 7.2e-6 at h = 0.15). In the presets' 12 Bohr box H₂ is box-clean only
// when nearly centred, so under the Estimated backend the app's own scenes get a box sized from
// the physics: L_est = extent + 2 (R_cloud + ROOM), never below the preset's own box nor
// [`ESTIMATED_MIN_BOX`]. extent: the largest nucleus–nucleus span along x, y or z; R_cloud: the
// largest, over the scene's species, distance at which that species' isolated-atom density (the
// app's own solve, `crate::cloud::AtomProfile`) falls below the box rule at the grid's h; ROOM:
// free room to move on each side. Nothing per element is stored (rule 1): R_cloud is read off the
// solver's answer for that Z, or the provisional bound below. The surrogate's shared collections
// are keyed by `surrogate::SurfaceKey` (functional, h, σ, tolerance, teacher, ionic surface — no
// `grid_n`), so a bigger box keeps them; a model *file*'s `Provenance` does carry `grid_n`, and
// the isolated-atom references (`sim::ReferenceKey`) are re-solved once per species for the new
// cell. Cost: a DFT solve at n_est³ costs ≈ (n_est / 81)³ of the Accurate one ([`scaled_cost`]),
// but most estimated steps are the model's, which do not scale with the grid.

/// Free room (Bohr) on each side of the molecule's span for it to move in before its cloud
/// reaches the box rule — a display/dynamics choice, the same for every element.
pub const ESTIMATED_ROOM: f64 = 2.0;

/// The estimated box is never smaller than the presets' box (Bohr).
pub const ESTIMATED_MIN_BOX: f64 = 12.0;

/// Within this distance (Bohr) of its cell's face a solved isolated-atom profile is pinched by
/// the Dirichlet wall (the confined tail falls as (1 − e^(−2κd))² of the free one, d the distance
/// to the wall: ≈ 5 % at d = 2.5 Bohr for κ ≈ 0.73, LSDA hydrogen), so the tail is read inside it.
pub const PROFILE_WALL_ZONE: f64 = 2.5;

/// The width (Bohr) of the trusted tail an exponential is fitted over when the profile's cell is
/// too small to show where the density falls below the box rule.
pub const PROFILE_FIT_WINDOW: f64 = 1.5;

/// **A provisional bound, not data:** R_cloud for a species whose isolated atom has not been
/// solved yet — where a hydrogenic 1s tail with Z_eff = 1, ρ = e^(−2r)/π, falls to the box rule
/// at h (5.35 Bohr at h = 0.15, 4.84 at 0.25; the same for every species). It is a stated estimate:
/// LSDA tails decay more slowly than the exact hydrogen's (κ = √(−2ε_HOMO) ≈ 0.73 for LSDA H) and
/// diffuse atoms such as Li more slowly still, so it can undersize the box. It is replaced by
/// [`profile_cloud_radius`] once the species' profile arrives (the scene note then says when the
/// box is short).
pub fn provisional_cloud_radius(h: f64) -> f64 {
    (1.0 / (std::f64::consts::PI * providers::box_affected_face_density(h))).ln() / 2.0
}

/// R_cloud from a solved isolated-atom profile: the radius (Bohr) where its density falls below
/// the box rule at spacing `h`. Only shells outside [`PROFILE_WALL_ZONE`] of the profile cell's
/// face are trusted; when the density crosses the rule there, the crossing is interpolated in
/// ln ρ between the shells around it, else an exponential is fitted (least squares in ln ρ) to the
/// last [`PROFILE_FIT_WINDOW`] of the trusted tail and extrapolated to the rule. `None` when the
/// profile cannot say (no shell above the rule, too few tail shells, or a tail that does not
/// decay): the caller uses [`provisional_cloud_radius`].
pub fn profile_cloud_radius(p: &crate::cloud::AtomProfile, h: f64) -> Option<f64> {
    let rule = providers::box_affected_face_density(h);
    let half = (p.n as f64 - 1.0) * p.h / 2.0;
    let r_t = (half - PROFILE_WALL_ZONE).min(*p.r.last()?);
    let tail: Vec<(f64, f64)> = p.r.iter().zip(&p.rho).map(|(&r, &v)| (r, v)).filter(|&(r, _)| r <= r_t).collect();
    let k = tail.iter().rposition(|&(_, v)| v >= rule)?;
    if let Some(&(r1, v1)) = tail.get(k + 1) {
        let (r0, v0) = tail[k];
        let f = if v1 > 0.0 { (v0 / rule).ln() / (v0 / v1).ln() } else { (v0 - rule) / (v0 - v1) };
        return Some(r0 + (r1 - r0) * f.clamp(0.0, 1.0));
    }
    let r_last = tail[k].0;
    let pts: Vec<(f64, f64)> = tail.iter().filter(|&&(r, v)| r >= r_last - PROFILE_FIT_WINDOW && v > 0.0).map(|&(r, v)| (r, v.ln())).collect();
    if pts.len() < 3 {
        return None;
    }
    let m = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
    let (mx, my) = (sx / m, sy / m);
    let (sxy, sxx) = pts.iter().fold((0.0, 0.0), |(a, b), &(x, y)| (a + (x - mx) * (y - my), b + (x - mx) * (x - mx)));
    let slope = sxy / sxx;
    if !(slope < 0.0) {
        return None;
    }
    Some(((rule.ln() - my) / slope + mx).max(r_last))
}

/// The largest nucleus–nucleus span along x, y or z (Bohr; 0 for one atom or none).
pub fn nuclear_extent(s: &Scene) -> f64 {
    (0..3)
        .map(|a| {
            let (lo, hi) = s.nuclei.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), n| (lo.min(n.pos[a]), hi.max(n.pos[a])));
            if hi > lo {
                hi - lo
            } else {
                0.0
            }
        })
        .fold(0.0, f64::max)
}

/// L_est = extent + 2 (R_cloud + [`ESTIMATED_ROOM`]) (Bohr; the floors are the caller's).
pub fn estimated_box_len(extent: f64, r_cloud: f64) -> f64 {
    extent + 2.0 * (r_cloud + ESTIMATED_ROOM)
}

/// The scene's R_cloud: the largest over its species.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudRadius {
    /// Bohr; 0 for a scene without nuclei.
    pub r: f64,
    /// The species it came from.
    pub z: Option<u8>,
    /// Some species had no usable profile: its [`provisional_cloud_radius`] stood in.
    pub provisional: bool,
}

/// The largest R_cloud over the scene's species at spacing `h`: from the profile of that Z solved
/// under the scene's ionic policy when `profiles` holds one ([`profile_cloud_radius`]), else the
/// provisional bound.
pub fn scene_cloud_radius(s: &Scene, h: f64, profiles: &[crate::cloud::AtomProfile]) -> CloudRadius {
    let mut out = CloudRadius { r: 0.0, z: None, provisional: false };
    let mut species: Vec<u8> = s.nuclei.iter().map(|n| n.z).collect();
    species.sort_unstable();
    species.dedup();
    for z in species {
        let solved = profiles.iter().filter(|p| p.z == z && p.ionic == s.ionic).find_map(|p| profile_cloud_radius(p, h));
        out.provisional |= solved.is_none();
        let r = solved.unwrap_or_else(|| provisional_cloud_radius(h));
        if out.z.is_none() || r > out.r {
            (out.r, out.z) = (r, Some(z));
        }
    }
    out
}

/// The estimated solver's box for a scene at spacing h ([`estimated_box`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimatedBox {
    pub extent: f64,
    pub cloud: CloudRadius,
    /// The box the rule asks for (Bohr): L_est, at least the base box and [`ESTIMATED_MIN_BOX`].
    pub want_len: f64,
    /// Its grid, [`grid_n`]`(want_len, h)` (capped at the largest grid).
    pub want_n: usize,
    /// The grid chosen: `want_n`, or the largest odd n down to the base box's that the GPU memory
    /// guard passes.
    pub n: usize,
    /// `n < want_n`: the memory guard (or the largest grid) cut it.
    pub capped: bool,
}

impl EstimatedBox {
    /// The chosen box length (Bohr).
    pub fn len(&self, h: f64) -> f64 {
        (self.n as f64 - 1.0) * h
    }
}

/// The estimated box for `s` at spacing `h` with base box `base_len` (the preset's or the file's
/// own: never shrunk): `fits(n)` is the GPU memory guard's verdict on an n³ grid; when the wanted
/// grid fails it, the largest odd n down to the base box's that passes (the base's own when none:
/// the guard then refuses it as before).
pub fn estimated_box(s: &Scene, base_len: f64, h: f64, profiles: &[crate::cloud::AtomProfile], fits: impl Fn(usize) -> bool) -> EstimatedBox {
    let extent = nuclear_extent(s);
    let cloud = scene_cloud_radius(s, h, profiles);
    let want_len = estimated_box_len(extent, cloud.r).max(base_len).max(ESTIMATED_MIN_BOX);
    let full = grid_n(want_len, h);
    let want_n = full.min(crate::scene::MAX_GRID_N);
    let floor = grid_n(base_len.max(ESTIMATED_MIN_BOX), h).min(want_n);
    let mut n = want_n;
    while n > floor && !fits(n) {
        n -= 2;
    }
    EstimatedBox { extent, cloud, want_len, want_n, n, capped: n < full }
}

/// The one-line accuracy hint per level, from `crates/validate/references.md` ("Quality presets":
/// H₂'s bond length R_e at σ = h on this solver; experiment 0.741 Å).
pub fn accuracy_hint(name: &str) -> &'static str {
    match name {
        "fast" => H2_FAST,
        "balanced" => "H₂ bond 0.947 Å (1.789 Bohr) at σ = h = 0.25; exact 0.741 Å",
        "accurate" => "H₂ bond 0.847 Å (1.601 Bohr) at σ = h = 0.15; exact 0.741 Å",
        "precise" => H2_PRECISE,
        _ => "custom h: not measured (Fast 0.30 … Precise 0.10 are); exact H₂ bond 0.741 Å",
    }
}

/// Measured by `crates/validate/tests/quality.rs` (`h2_bond_length_per_quality`, report-only;
/// `references.md` "Quality presets"). Balanced and Accurate there reproduce the M4 pin (1.78894)
/// and the oracle scan's R_e(σ = 0.15) (1.6008).
const H2_FAST: &str = "H₂ bond 1.002 Å (1.893 Bohr) at σ = h = 0.30; exact 0.741 Å";
const H2_PRECISE: &str = "H₂ bond 0.807 Å (1.524 Bohr) at σ = h = 0.10; exact 0.741 Å";

#[cfg(test)]
mod tests {
    use super::*;

    /// The budget's order (user request 2026-10-19): the variable, else half the card's reported
    /// memory, else 12 GB; the refusal names the budget and its source.
    #[test]
    fn the_vram_budget_follows_the_variable_then_the_card_then_the_default() {
        let card = 25_753_026_560; // 24 GiB, as the RX 7900 XTX's Vulkan heap reports it
        assert_eq!(budget_from(Some(8.0), card), VramBudget { bytes: 8_000_000_000, source: BudgetSource::Env });
        let b = budget_from(None, card);
        assert_eq!(b, VramBudget { bytes: card / 2, source: BudgetSource::Adapter { card_bytes: card } });
        assert_eq!((b.gb(), b.why()), ("12.9 GB".into(), "50 % of this GPU's 25.8 GB; ATOMIC_SANDBOX_VRAM_GB overrides it".into()));
        let d = budget_from(None, 0);
        assert_eq!((d.bytes, d.source), (12_000_000_000, BudgetSource::Default));
        assert!(d.why().contains("did not report"));
        let mut big = Scene::preset("h2").unwrap();
        big.grid_n = 201;
        let e = vram_check(&big, 2, budget_from(None, 4_000_000_000)).unwrap_err();
        assert!(e.contains("over the 2.0 GB budget (50 % of this GPU's 4.0 GB; ATOMIC_SANDBOX_VRAM_GB overrides it)"), "{e}");
    }

    /// The presets' 12 Bohr box at every level: 41³, 49³, 81³, 121³ — never smaller than 12;
    /// h2o's 14 Bohr box keeps its extent too.
    #[test]
    fn a_quality_change_keeps_the_box() {
        let want = [("fast", 41), ("balanced", 49), ("accurate", 81), ("precise", 121)];
        for (name, n) in want {
            let s = preset("h2", parse(name).unwrap()).unwrap();
            assert_eq!(s.grid_n, n, "{name}");
            assert!(box_len(&s) >= 12.0 - 1e-9, "{name}: box {}", box_len(&s));
            s.validate().unwrap();
        }
        let w = preset("h2o", DEFAULT_H).unwrap();
        assert_eq!((w.grid_n, w.grid_h), (95, 0.15));
        assert!(box_len(&w) >= 14.0 - 1e-9);
        // custom: rounded up to whole odd n, the box at least as long
        for h in [0.08, 0.123, 0.2, 0.35] {
            let s = regrid(&Scene::preset("h").unwrap(), 12.0, h);
            assert_eq!(s.grid_n % 2, 1);
            assert!(box_len(&s) >= 12.0 - 1e-9 && (s.grid_n as f64 - 3.0) * s.grid_h < 12.0, "h {h}: n {}", s.grid_n);
            s.validate().unwrap();
        }
    }

    /// σ follows h unless it was set: `regrid` never touches it.
    #[test]
    fn sigma_follows_h_unless_set() {
        let s = preset("h2", 0.30).unwrap();
        assert_eq!(s.sigma, None, "σ = h");
        let mut set = s.clone();
        set.sigma = Some(0.2);
        assert_eq!(regrid(&set, 12.0, 0.15).sigma, Some(0.2));
    }

    /// Custom snaps to 3 decimals within [0.08, 0.35]; the levels parse by name.
    #[test]
    fn custom_snaps_and_levels_parse() {
        assert_eq!(snap(0.123456), 0.123);
        assert_eq!(snap(0.1235), 0.124);
        assert_eq!(snap(0.01), H_MIN);
        assert_eq!(snap(9.0), H_MAX);
        assert_eq!(snap(f64::NAN), DEFAULT_H);
        assert_eq!(regrid(&Scene::preset("h").unwrap(), 12.0, 0.1504).grid_h, 0.15);
        assert_eq!(parse("Accurate"), Some(0.15));
        assert_eq!(parse("fast"), Some(0.30));
        assert_eq!(parse("precise"), Some(0.10));
        assert_eq!(parse("balanced"), Some(0.25));
        assert_eq!(parse("ultra"), None);
        assert_eq!(level_of(0.15), Some("accurate"));
        assert_eq!(level_of(0.2), None);
        assert_eq!(title("accurate"), "Accurate");
        assert!((scaled_cost(1.0, 49, 81) - (81.0f64 / 49.0).powi(3)).abs() < 1e-12);
    }

    // ---- the estimated solver's box ----

    /// A species' isolated-atom profile as the app makes it (`cloud::radial_profile`) from the
    /// density ρ(r) = κ³/π e^(−2κr) (a 1s-like tail of decay κ, one electron) on a centred n³ grid of
    /// spacing h, pinched by the Dirichlet wall one h beyond the outermost plane as a confined tail
    /// is: × (1 − e^(−2κd))², d the distance to the nearest wall.
    fn tail_profile(z: u8, kappa: f64, n: usize, h: f64) -> crate::cloud::AtomProfile {
        let half = (n as f64 - 1.0) * h / 2.0;
        let mut rho = vec![0.0f32; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = [i, j, k].map(|c| -half + c as f64 * h);
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    let d = half + h - p.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                    let free = kappa.powi(3) / std::f64::consts::PI * (-2.0 * kappa * r).exp();
                    rho[i + n * (j + n * k)] = (free * (1.0 - (-2.0 * kappa * d).exp()).powi(2)) as f32;
                }
            }
        }
        crate::cloud::radial_profile(z, &rho, n, h, [-half; 3], None, sim::IonicPolicy::None)
    }

    /// Where κ³/π e^(−2κr) falls to the box rule at h (Bohr): the exact answer the profile is read for.
    fn exact_radius(kappa: f64, h: f64) -> f64 {
        (kappa.powi(3) / (std::f64::consts::PI * providers::box_affected_face_density(h))).ln() / (2.0 * kappa)
    }

    /// H₂ along x with R = 1.4 Bohr on the presets' 12 Bohr box at h, all-electron.
    fn h2(h: f64) -> Scene {
        let mut s = Scene::preset("h2").unwrap();
        for (n, x) in s.nuclei.iter_mut().zip([0.7, -0.7]) {
            n.pos = [x, 0.0, 0.0];
        }
        regrid(&s, 12.0, h)
    }

    /// The provisional bound is the hydrogenic 1s tail (Z_eff = 1) at the box rule: 5.35 Bohr at
    /// h = 0.15 (rule 7.2e-6), 4.84 at 0.25 (2e-5) — the same for every species.
    #[test]
    fn the_provisional_cloud_radius_is_the_hydrogenic_tail_at_the_box_rule() {
        assert!((provisional_cloud_radius(0.15) - 5.348).abs() < 1e-3, "{}", provisional_cloud_radius(0.15));
        assert!((provisional_cloud_radius(0.25) - 4.837).abs() < 1e-3, "{}", provisional_cloud_radius(0.25));
        assert!((provisional_cloud_radius(0.15) - exact_radius(1.0, 0.15)).abs() < 1e-12);
    }

    /// R_cloud from a solved profile: in the presets' 12 Bohr cell the hydrogen tail never reaches
    /// the rule inside the trusted zone, so it is extrapolated from the trusted tail — the exact
    /// hydrogen (κ = 1) to 5.35 Bohr and an LSDA-like tail (κ = √(2 × 0.269), the LSDA H atom's
    /// HOMO −0.269 Ha) to 6.66, as a 20 Bohr cell reads it; a fast tail (κ = 2) crosses inside the
    /// trusted zone and is interpolated; a tail that does not decay, or lies below the rule, says
    /// nothing.
    #[test]
    fn the_cloud_radius_is_read_off_the_solved_profile() {
        let lsda = (2.0f64 * 0.269).sqrt();
        for (kappa, n, h, tol) in [(1.0, 81, 0.15, 0.05), (lsda, 81, 0.15, 0.25), (lsda, 49, 0.25, 0.25), (lsda, 135, 0.15, 0.1), (2.0, 81, 0.15, 0.05)] {
            let p = tail_profile(1, kappa, n, h);
            let r = profile_cloud_radius(&p, 0.15).unwrap();
            let want = exact_radius(kappa, 0.15);
            assert!((r - want).abs() < tol, "κ {kappa:.3}, {n}³ × {h}: R_cloud {r:.3}, exact {want:.3}");
        }
        assert!((exact_radius(lsda, 0.15) - 6.66).abs() < 0.01, "{}", exact_radius(lsda, 0.15));
        let mut flat = tail_profile(1, 1.0, 41, 0.3);
        flat.rho.iter_mut().for_each(|v| *v = 1e-3);
        assert_eq!(profile_cloud_radius(&flat, 0.15), None, "a tail that does not decay");
        let mut thin = flat.clone();
        thin.rho.iter_mut().for_each(|v| *v = 1e-9);
        assert_eq!(profile_cloud_radius(&thin, 0.15), None, "nothing above the rule");
    }

    /// The extent is the largest span along an axis (not the diagonal), 0 for a lone atom.
    #[test]
    fn the_extent_is_the_largest_span_along_an_axis() {
        assert!((nuclear_extent(&h2(0.15)) - 1.4).abs() < 1e-12);
        assert_eq!(nuclear_extent(&Scene::preset("h").unwrap()), 0.0);
        assert_eq!(nuclear_extent(&Scene::preset("empty").unwrap()), 0.0);
        let mut s = h2(0.15);
        s.nuclei[1].pos = [-0.7, 2.5, -1.0];
        assert!((nuclear_extent(&s) - 2.5).abs() < 1e-12, "y spans 2.5, x 1.4, z 1.0");
    }

    /// H₂ (one species, extent 1.4 Bohr) at h = 0.15: before H's profile, R_cloud is the
    /// provisional 5.35 Bohr → L_est = 1.4 + 2 (5.35 + 2) = 16.1 Bohr → 109³ (16.2 Bohr); with an
    /// LSDA-like solved profile R_cloud ≈ 6.66 → L_est ≈ 18.7 → 127³. Both above the 12 Bohr floor.
    #[test]
    fn h2_gets_a_box_sized_from_its_cloud() {
        let s = h2(0.15);
        let eb = estimated_box(&s, 12.0, 0.15, &[], |_| true);
        assert!(eb.cloud.provisional && eb.cloud.z == Some(1));
        assert!((eb.want_len - (1.4 + 2.0 * (provisional_cloud_radius(0.15) + ESTIMATED_ROOM))).abs() < 1e-12);
        assert!((eb.want_len - 16.097).abs() < 1e-3, "{}", eb.want_len);
        assert_eq!((eb.want_n, eb.n, eb.capped), (109, 109, false));
        assert!(eb.len(0.15) >= eb.want_len && eb.len(0.15) >= 12.0);
        let p = tail_profile(1, (2.0f64 * 0.269).sqrt(), 81, 0.15);
        let eb = estimated_box(&s, 12.0, 0.15, std::slice::from_ref(&p), |_| true);
        assert!(!eb.cloud.provisional);
        assert!(eb.want_len >= 12.0 && (eb.cloud.r - 6.66).abs() < 0.25, "R_cloud {}", eb.cloud.r);
        assert_eq!(eb.n, grid_n(eb.want_len, 0.15));
        assert!((123..=129).contains(&eb.n), "{}", eb.n);
        // a profile solved under another ionic policy is not this scene's
        let other = Scene { ionic: sim::IonicPolicy::Accepted, ..s.clone() };
        assert!(estimated_box(&other, 12.0, 0.15, std::slice::from_ref(&p), |_| true).cloud.provisional);
        // never below the floor, nor the base box (h2o's 14)
        let tiny = Scene { nuclei: s.nuclei[..1].to_vec(), ..s.clone() };
        let fast = tail_profile(1, 3.0, 81, 0.15);
        assert_eq!(estimated_box(&tiny, 12.0, 0.15, std::slice::from_ref(&fast), |_| true).n, 81);
        assert_eq!(estimated_box(&tiny, 14.0, 0.15, std::slice::from_ref(&fast), |_| true).n, 95);
    }

    /// A lone atom: extent 0, L_est = 2 (R_cloud + 2) — 14.7 Bohr (99³) at h = 0.15 provisionally.
    #[test]
    fn a_lone_atom_gets_twice_its_cloud_and_room() {
        let s = regrid(&Scene::preset("h").unwrap(), 12.0, 0.15);
        let eb = estimated_box(&s, 12.0, 0.15, &[], |_| true);
        assert_eq!(eb.extent, 0.0);
        assert!((eb.want_len - 2.0 * (provisional_cloud_radius(0.15) + 2.0)).abs() < 1e-12);
        assert_eq!(eb.n, 99);
    }

    /// Several species: the largest R_cloud wins, a species without a profile stands in at the
    /// provisional bound and flags the answer provisional.
    #[test]
    fn several_species_take_the_largest_cloud() {
        let mut s = h2(0.15);
        s.nuclei[1].z = 3;
        let tight_h = tail_profile(1, 2.0, 81, 0.15);
        let diffuse_li = tail_profile(3, 0.6, 81, 0.15);
        let c = scene_cloud_radius(&s, 0.15, &[tight_h.clone(), diffuse_li.clone()]);
        assert_eq!((c.z, c.provisional), (Some(3), false));
        // a diffuse tail in the 12 Bohr cell: the wall still steepens the fitted window by ≈ 9 %
        // (2 · 2κ e^(−2κd) at d ≈ 2.65 Bohr, κ = 0.6), so R_cloud reads ≈ 0.35 Bohr short of 7.64;
        // the profile solved in the grown box reads it closer
        assert!((c.r - exact_radius(0.6, 0.15)).abs() < 0.5, "{c:?}");
        let c = scene_cloud_radius(&s, 0.15, std::slice::from_ref(&tight_h));
        assert_eq!((c.z, c.provisional), (Some(3), true), "Li unsolved: the provisional 5.35 beats H's 3.2");
        assert!((c.r - provisional_cloud_radius(0.15)).abs() < 1e-12);
        let c = scene_cloud_radius(&s, 0.15, &[diffuse_li]);
        assert!(c.provisional && c.z == Some(3), "H unsolved: still flagged");
        assert_eq!(scene_cloud_radius(&Scene::preset("empty").unwrap(), 0.15, &[]), CloudRadius { r: 0.0, z: None, provisional: false });
    }

    /// The memory guard: when the wanted grid does not fit, the largest odd n that does, down to
    /// the base box's (then the guard refuses as before); the largest grid caps it too.
    #[test]
    fn the_estimated_box_falls_back_to_the_largest_grid_that_fits() {
        let s = h2(0.15);
        let eb = estimated_box(&s, 12.0, 0.15, &[], |n| n <= 101);
        assert_eq!((eb.want_n, eb.n, eb.capped), (109, 101, true));
        let eb = estimated_box(&s, 12.0, 0.15, &[], |_| false);
        assert_eq!((eb.n, eb.capped), (81, true), "nothing fits: the base box's grid, which the guard then refuses");
        let mut far = s.clone();
        far.nuclei[1].pos = [-40.0, 0.0, 0.0];
        let eb = estimated_box(&far, 12.0, 0.15, &[], |_| true);
        assert_eq!((eb.want_n, eb.n, eb.capped), (crate::scene::MAX_GRID_N, crate::scene::MAX_GRID_N, true));
        // with a budget the check is the app's guard (2 solvers: teacher + auditor)
        let budget = VramBudget::fixed(u64::MAX);
        let eb = estimated_box(&s, 12.0, 0.15, &[], |n| vram_check(&Scene { grid_n: n, ..s.clone() }, 2, budget).is_ok());
        assert_eq!(eb.n, 109);
        let eb = estimated_box(&s, 12.0, 0.15, &[], |n| vram_check(&Scene { grid_n: n, ..s.clone() }, 2, VramBudget::fixed(1)).is_ok());
        assert_eq!((eb.n, eb.capped), (81, true));
    }
}
