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

/// The GPU memory budget (GB) when `ATOMIC_SANDBOX_VRAM_GB` is not set: wgpu reports no total
/// VRAM, so this is a documented constant — half the project's 24 GB RX 7900 XTX, which leaves room
/// for the renderer, the driver and other applications (`providers::ks`, "GPU memory"). Set the
/// variable on a smaller (or larger) card.
pub const VRAM_BUDGET_GB: f64 = 12.0;

/// The GPU memory budget in bytes: `ATOMIC_SANDBOX_VRAM_GB` (a positive number of GB) or
/// [`VRAM_BUDGET_GB`].
pub fn vram_budget_bytes() -> u64 {
    let gb = std::env::var("ATOMIC_SANDBOX_VRAM_GB").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|x| x.is_finite() && *x > 0.0);
    (gb.unwrap_or(VRAM_BUDGET_GB) * 1e9) as u64
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

/// `Err(message)` when `scene` on `solvers` GPU solvers would exceed `budget` bytes (or has an
/// element without a mass, which the worker refuses anyway: no estimate then).
pub fn vram_check(scene: &Scene, solvers: u64, budget: u64) -> Result<(), String> {
    let Ok(s) = scene.to_sim() else { return Ok(()) };
    let need = solvers * providers::gpu_footprint_bytes(&s);
    if solvers > 0 && need > budget {
        return Err(format!(
            "refused: {}³ × {:.3} Bohr needs ≈ {:.1} GB of GPU memory ({} solver{}, {} electrons), over the {:.1} GB budget (ATOMIC_SANDBOX_VRAM_GB) — choose a coarser quality or fewer atoms",
            scene.grid_n,
            scene.grid_h,
            need as f64 / 1e9,
            solvers,
            if solvers == 1 { "" } else { "s: teacher + auditor" },
            s.electrons().max(0),
            budget as f64 / 1e9
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
}
