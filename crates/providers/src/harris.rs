//! The input density of a Harris display solve, and its GPU memory (live-orbitals spec §4.1–4.2;
//! `docs/superpowers/specs/2026-11-14-live-orbitals-design.md`). `crate::ks::GpuKs::harris_display`
//! builds ρ_in here on the CPU and hands it to `qm_gpu::GpuSolver::harris`.
//!
//! # ρ_in (spec §4.1; atomic units: Bohr, e/Bohr³)
//! Each nucleus A (species Z_A at R_A) contributes its species' isolated-atom radial profile
//! p_A(r) = `sim::AtomProfile::at` — the spherically averaged density of this program's own
//! isolated-atom solve on the scene's surface (grid, σ, ionic policy) — splatted onto the grid
//! nodes r_i within the profile's reach, in f64:
//!
//! ```text
//!   ρ̃_A(r_i) = p_A(|r_i − R_A|)        for |r_i − R_A| ≤ reach_A, else 0
//!   S_A      = h³ Σ_i ρ̃_A(r_i)         (Neumaier; the atom's on-grid electrons: `RhoIn::per_atom`)
//!   ρ_A      = ρ̃_A · c_A / S_A          (renormalised: h³ Σ ρ_A = c_A exactly, up to rounding)
//!   C        = Σ_A c_A,   N = n↑ + n↓ = C − charge
//!   s        = N / C                     (`RhoIn::rescale`; 1 for a neutral scene)
//!   ρ_in,σ   = s · Σ_A ρ_A · n_σ / N     ⇒  h³ Σ ρ_in,σ = n_σ,  h³ Σ ρ_in = N
//! ```
//!
//! with c_A the electrons the atom brings: Z_A all-electron, Z_v = `IonicSetup::valence_charge`
//! under a pseudopotential (the caller passes them as `counted`). Renormalising each atom first
//! removes the quadrature and placement error of splatting a profile solved at a grid point onto
//! a nucleus that sits between grid points (and the part of the profile beyond a near face); the
//! rescale then puts a charged scene's electron count on the superposition (H₃⁺: C = 3, N = 2,
//! s = 2/3), which the payload states (`OrbitalKind::Harris::rescale`, and a note). The spin split
//! is proportional, ρ_in,σ = ρ_in · n_σ/N (n_σ from the scene's multiplicity rule): superposed atoms
//! carry no molecular spin, a decided simplification (spec §4.1).
//!
//! # The partial core (spec §4.1)
//! ρ_in is the **valence** density only: the profiles must be solved without the display's ρ_pc
//! (the auditor's Atom jobs, live-orbitals Task 5, which pins it by a test). ρ_pc never enters
//! ρ_in's Hartree term; `GpuSolver::harris` adds ½ρ_pc to each spin's xc input (NLCC), as the SCF
//! does. Nothing here can detect a profile that carries ρ_pc — the renormalisation to Z_v would
//! scale it down silently — so that rule is the caller's.
//!
//! # GPU memory ([`harris_footprint_bytes`]; spec §4.2 "VRAM")
//! What a Harris solve holds on top of the solve (`crate::ks::solve_footprint_bytes`) and the
//! orbital view (`crate::ks::orbital_view_footprint_bytes`) of the same n, n_σ and m, counted
//! from `qm_gpu::harris`'s "Fields and VRAM":
//! - the block per channel at its growth bound, n_σ + m + G + W vectors (G = `UNOCC_GUARD`; 0 for a
//!   channel with n_σ + m = 0): the state's own vectors, a block grows behind its Ritz vectors (a
//!   shared shell or a short window) up to that bound, and the fields it trims stay on the state's
//!   free list. W = `MAX_WITNESS_GROWTH` under an ionic setup, 0 without: the occupations are
//!   `DegenerateShells` exactly when there is a setup (`crate::ks::problem`), and `Aufbau` has no
//!   window and n_occ = n_σ, so an all-electron block never grows (as `crate::ks::pseudo_extra_bytes`
//!   counts the SCF's witness growth only for a pseudo solve);
//! - 13 fields: ρ_in↑↓, ρ, V_H, v_nuc, v↑↓, exc, v_xc↑↓, ρ_out↑↓ and one scratch; + 3 under an ionic
//!   setup (ρ_pc and the two xc inputs ρ_σ + ½ρ_pc);
//! - the solver's one `LobpcgWork` (8 fields per vector) grown to the largest block, max n_σ + m +
//!   G + W: 8(m + G + W) over the SCF's 8·max n_σ, less the view's own growth of the same work
//!   space (`OrbitalView::work_growth_fields(m)` = 8(m + G) for m > 0), which the view's term counts;
//! - × 1.2, the slack of the other terms (uniforms, readback buffers, allocator slack). The Harris
//!   state's own `GpuIonic` (local tables, nonlocal point lists: ≈ (4 + 4·rows) bytes per point in
//!   each projector ball, ~0.1 MB for a pseudo-O at h = 0.15) is inside that slack: 13 fields at
//!   97³ are 60 MB, whose 20 % covers ~100 such ions.
//!
//! Each field is `crate::ks::field_bytes(n)`.
use crate::ks::field_bytes;
use qm::orbitals::UNOCC_GUARD;
use qm::scf::MAX_WITNESS_GROWTH;
use qm_gpu::orbitals::OrbitalView;
use sim::AtomProfile;

/// The Harris solve's input density (module doc): ρ_in,↑ and ρ_in,↓ on the grid (`qm::Grid`
/// layout, e/Bohr³), the factor s = N/C the renormalised superposition was scaled by, and each
/// atom's on-grid electrons S_A before its renormalisation (scene order; the debug readout's).
#[derive(Clone, Debug, PartialEq)]
pub struct RhoIn {
    pub up: Vec<f64>,
    pub dn: Vec<f64>,
    pub rescale: f64,
    pub per_atom: Vec<f64>,
}

/// ρ_in of `nuclei` on `grid` (module doc, "ρ_in"): each nucleus's species profile from `profiles`
/// (the first with its Z; the caller passes only profiles valid for the scene's surface),
/// renormalised to `counted[A]` electrons, summed, rescaled to N = `n_up + n_dn` and split n_σ/N.
/// `Err` (a sentence for the payload's note) when N = 0, when a species has no profile, or when an
/// atom puts no density on the grid (S_A ≤ 0). Panics if `counted` is not one per nucleus.
pub fn superposition(grid: &qm::Grid, nuclei: &[qm::Nucleus], profiles: &[AtomProfile], counted: &[f64], n_up: usize, n_dn: usize)
                     -> Result<RhoIn, String> {
    assert_eq!(counted.len(), nuclei.len(), "superposition: one electron count per nucleus");
    let n_el = n_up + n_dn;
    if n_el == 0 {
        return Err("no electrons: the scene's charge leaves none, so there are no orbitals to solve".into());
    }
    let dv = grid.dv();
    let mut sum = vec![0.0f64; grid.len()];
    let mut per_atom = Vec::with_capacity(nuclei.len());
    let (mut at, mut val) = (Vec::new(), Vec::new());
    for (a, nuc) in nuclei.iter().enumerate() {
        let p = profiles.iter().find(|p| p.z == nuc.z).ok_or_else(|| {
            format!("no isolated-atom profile for {} on this grid yet: the superposed density needs one per species", units::elements::symbol(nuc.z))
        })?;
        splat(grid, nuc.pos, p, &mut at, &mut val);
        let s_a = qm_gpu::reduce::neumaier(val.iter().copied()) * dv;
        if !(s_a > 0.0) {
            return Err(format!("atom {a} ({}) puts no density on the grid (its profile's reach ends before the nearest node)", units::elements::symbol(nuc.z)));
        }
        per_atom.push(s_a);
        let scale = counted[a] / s_a;
        for (&i, &v) in at.iter().zip(&val) {
            sum[i] += scale * v;
        }
    }
    let rescale = n_el as f64 / qm_gpu::reduce::neumaier(counted.iter().copied());
    let split = |n_s: usize| -> Vec<f64> {
        let f = rescale * (n_s as f64 / n_el as f64);
        sum.iter().map(|v| f * v).collect()
    };
    Ok(RhoIn { up: split(n_up), dn: split(n_dn), rescale, per_atom })
}

/// The nodes of `grid` within `p`'s reach of `pos` (their indices into `at`) and p(|r − pos|) at
/// each (into `val`; both cleared first).
fn splat(grid: &qm::Grid, pos: [f64; 3], p: &AtomProfile, at: &mut Vec<usize>, val: &mut Vec<f64>) {
    at.clear();
    val.clear();
    let (h, o, reach) = (grid.h, grid.origin, p.reach());
    // the index range [lo, hi] of nodes within `reach` of pos along axis a (empty when lo > hi)
    let range = |a: usize| -> (usize, usize) {
        let lo = ((pos[a] - reach - o[a]) / h).ceil().max(0.0);
        let hi = ((pos[a] + reach - o[a]) / h).floor().min(grid.n[a] as f64 - 1.0);
        if hi < lo { (1, 0) } else { (lo as usize, hi as usize) }
    };
    let ((x0, x1), (y0, y1), (z0, z1)) = (range(0), range(1), range(2));
    if x0 > x1 || y0 > y1 || z0 > z1 {
        return;
    }
    for k in z0..=z1 {
        let dz = o[2] + k as f64 * h - pos[2];
        for j in y0..=y1 {
            let dy = o[1] + j as f64 * h - pos[1];
            for i in x0..=x1 {
                let dx = o[0] + i as f64 * h - pos[0];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r <= reach {
                    at.push(grid.idx(i, j, k));
                    val.push(p.at(r));
                }
            }
        }
    }
}

/// The GPU memory (bytes) a Harris display solve holds on top of the solve and the orbital view
/// of an n³ grid with n↑, n↓ occupied orbitals and m unoccupied ones per channel (module doc, "GPU
/// memory"); `pseudised`: the scene has an ionic setup (the partial-core fields and the witness
/// growth counted).
pub fn harris_footprint_bytes(n: usize, n_up: usize, n_dn: usize, m: usize, pseudised: bool) -> u64 {
    // the witness growth only under a setup: `DegenerateShells` exactly then (`crate::ks::problem`)
    let (g, w) = (UNOCC_GUARD, if pseudised { MAX_WITNESS_GROWTH } else { 0 });
    let block = |n_s: usize| if n_s + m > 0 { n_s + m + g + w } else { 0 };
    let vectors = block(n_up) + block(n_dn);
    let fields = if pseudised { 13 + 3 } else { 13 };
    // the one `LobpcgWork` grown to the largest block, less what the view's term counts of it
    let work = 8 * (m + g + w) - OrbitalView::work_growth_fields(m);
    ((vectors + fields + work) as u64 as f64 * field_bytes(n) as f64 * 1.2) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use qm::{Grid, Nucleus};
    use sim::{radial_profile, IonicPolicy};

    /// The hydrogen 1s density e^{−2r}/π at the centre of an n³ grid (the app cloud's test atom).
    fn h1s_profile(n: usize, h: f64) -> AtomProfile {
        let grid = Grid::cube(n, h);
        let rho: Vec<f32> = grid.sample(|p| (-2.0 * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).exp() / std::f64::consts::PI)
            .into_iter().map(|v| v as f32).collect();
        radial_profile(1, &rho, n, h, grid.origin, None, IonicPolicy::None)
    }

    fn integral(grid: &Grid, f: &[f64]) -> f64 {
        grid.integrate(f)
    }

    /// Spec §4.1 / Review Focus 3: two analytic 1s profiles at off-grid positions on 61³ × 0.2.
    /// counted [1, 1], n = (1, 1): h³Σρ_σ = 1 each to 1e-10 and the rescale is 1 — the profile alone
    /// carries ≈ 0.99 on the grid (`per_atom`), the renormalisation makes it exact. counted [6, 1] (a
    /// pseudo-O-like valence count on the first), n = (4, 3): h³Σρ = 7 to 1e-10, rescale 1. A charged
    /// pair (n = (1, 0)) is rescaled by 1/2. And the three refusals: N = 0, a species without a
    /// profile, and an atom whose profile puts nothing on the grid.
    #[test]
    fn superposition_integrates_to_the_valence_count() {
        let (n, h) = (61usize, 0.2);
        let p = h1s_profile(n, h);
        let grid = Grid::cube(n, h);
        let nuclei = [Nucleus { pos: [-0.73, 0.05, 0.0], z: 1 }, Nucleus { pos: [0.71, -0.03, 0.02], z: 1 }];
        let profiles = std::slice::from_ref(&p);

        let r = superposition(&grid, &nuclei, profiles, &[1.0, 1.0], 1, 1).expect("H has a profile");
        let (up, dn) = (integral(&grid, &r.up), integral(&grid, &r.dn));
        println!("H + H: S_A = {:?} (profile carries {:.6}), ∫ρ↑ = {up:.15}, ∫ρ↓ = {dn:.15}, rescale {}", r.per_atom, p.electrons, r.rescale);
        assert!((up - 1.0).abs() < 1e-10 && (dn - 1.0).abs() < 1e-10, "{up} {dn}");
        assert_eq!(r.rescale, 1.0);
        assert_eq!(r.per_atom.len(), 2);
        assert!(r.per_atom.iter().all(|s| (s - 1.0).abs() < 2e-2 && *s != 1.0), "the bare splat is not exact: {:?}", r.per_atom);
        assert!(r.up.iter().zip(&r.dn).all(|(a, b)| a == b), "n↑ = n↓: the same split");

        let r = superposition(&grid, &nuclei, profiles, &[6.0, 1.0], 4, 3).expect("H has a profile");
        let (up, dn) = (integral(&grid, &r.up), integral(&grid, &r.dn));
        println!("valence 6 + 1: ∫ρ = {:.15} (↑ {up:.12}, ↓ {dn:.12}), rescale {}", up + dn, r.rescale);
        assert!((up + dn - 7.0).abs() < 1e-10, "{}", up + dn);
        assert!((up - 4.0).abs() < 1e-10 && (dn - 3.0).abs() < 1e-10, "{up} {dn}");
        assert_eq!(r.rescale, 1.0);

        // a cation of the pair: C = 2, N = 1, s = 1/2
        let r = superposition(&grid, &nuclei, profiles, &[1.0, 1.0], 1, 0).expect("H has a profile");
        assert!((r.rescale - 0.5).abs() < 1e-15, "{}", r.rescale);
        assert!((integral(&grid, &r.up) - 1.0).abs() < 1e-10 && r.dn.iter().all(|&v| v == 0.0));

        let e = superposition(&grid, &nuclei[..1], profiles, &[1.0], 0, 0).expect_err("N = 0");
        assert!(e.contains("no electrons"), "{e}");
        let e = superposition(&grid, &[Nucleus { pos: [0.0; 3], z: 8 }], profiles, &[6.0], 3, 3).expect_err("no O profile");
        assert!(e.contains("O") && e.contains("profile"), "{e}");
        let far = [Nucleus { pos: [40.0, 0.0, 0.0], z: 1 }];
        let e = superposition(&grid, &far, profiles, &[1.0], 1, 0).expect_err("nothing on the grid");
        assert!(e.contains("grid"), "{e}");
    }

    /// The memory term's field count (module doc, "GPU memory"), at the counts it is built from:
    /// H₂ all-electron, m = 0: blocks 2 × (1 + 2) + 13 + the work's 8·2 = 35 fields; m = 4: blocks
    /// 2 × 7 + 13 + (8·6 − the view's 8·6) = 27; pseudo-O triplet (n = 4, 2) at m = 2: blocks
    /// (4 + 2 + 2 + 16) + (2 + 2 + 2 + 16) + 16 + (8·20 − 8·4) = 46 + 16 + 128 = 190; an empty
    /// channel (H atom ↓, m = 0) holds no block.
    #[test]
    fn harris_footprint_counts_its_fields() {
        let fields = |b: u64| b as f64 / (field_bytes(33) as f64 * 1.2);
        let at = |n_up, n_dn, m, ps| fields(harris_footprint_bytes(33, n_up, n_dn, m, ps));
        for (got, want) in [(at(1, 1, 0, false), 35.0), (at(1, 1, 4, false), 27.0), (at(4, 2, 2, true), 190.0), (at(1, 0, 0, false), 3.0 + 13.0 + 16.0)] {
            assert!((got - want).abs() < 1e-3, "{got} fields, expected {want} (the bytes are truncated to an integer)");
        }
    }
}
