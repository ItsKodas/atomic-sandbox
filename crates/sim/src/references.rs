//! Isolated-atom references (M6 spec §3.2): the energy and the **shell-binned radial density**
//! ρ̄_Z(r) of the isolated neutral atom of species Z, solved by this solver at the scene's own
//! grid and σ. The promolecule of the bond detector (`crate::bonds`) is the sum of these tables
//! over every nucleus; the surrogate's sampling pass uses the energies (ruling S9). Nothing here
//! is tabulated per element (CLAUDE.md rule 1): every entry comes from a solve, and the only
//! inputs are Z and the mass.
//!
//! # Equations (atomic units: Bohr, Hartree; e/Bohr³)
//!
//! **The scan (ruling S9).** One nucleus at the origin of the scene's cell, charge 0, solved at
//! every allowed multiplicity `allowed_multiplicities(Z)` (2S + 1 of the parity of Z, from the
//! lowest up to 5, 2S ≤ Z), each solve cold (`reset()` before it: another spin state's orbitals
//! must not steer the SCF); the lowest energy wins and its density is the reference's.
//!
//! **Shell binning (MUST-1).** After a solve the provider is asked for ρ = ρ↑ + ρ↓ at every grid
//! node within r_max = half-extent − 3h of the origin ([`shell_nodes`]; node values are exact —
//! trilinear interpolation returns a node's stored value, so no interpolation bias enters the
//! table). The nodes are binned by r = |node| in bins of width dr = h/2, and each non-empty bin
//! contributes one entry
//!
//! ```text
//! r̄_b = mean of r over the bin's nodes,   ℓ_b = mean of ln ρ over the bin's nodes
//! ```
//!
//! (values ≤ 0 are clamped to 1e-300 before the log). The geometric mean rather than the brief's
//! arithmetic one, by this estimate: for an exponential tail ρ = C e^{−2κr} the log-linear
//! interpolant through (r̄_b, ℓ_b) is **exact**, while an arithmetic-mean entry (r̄_b, ln mean ρ)
//! sits above ρ(r̄_b) by ≈ Var_b(r)·(2κ)²/2 — with Var_b(r) ≈ dr²/12 = 1.3e-3 Bohr² that is
//! 0.26 % for hydrogen (κ = 1) and 0.74 % for helium (2κ = 2Z_eff = 3.375 with Z_eff = 1.6875;
//! 0.47 % for its asymptotic 2√(2I) = 2.68 tail), a quarter of the trilinear bias the
//! same-operator construction was built to remove. For a Gaussian the residual of the
//! geometric mean is Var_b(r)/(2w²) ≈ 0.1 % at w = 0.8 (measured in the tests below).
//! Empty bins are simply absent (there is no node between r = 0 and r = h); r = 0 is the centre
//! node alone for an odd-n cell.
//!
//! **Evaluation.** [`RadialDensity::at`] interpolates ln ρ linearly in r between neighbouring
//! entries — exact for an exponential tail, second order in the local change of the log-slope
//! otherwise — except on the first segment when the table starts at r = 0, where ln ρ is linear
//! in **r²**: a smeared nucleus makes ρ even in r with zero slope at the origin, ρ(r) = ρ₀ +
//! ½ρ₀″ r² + …, and a Gaussian-like ln ρ = ln ρ₀ − r²/2w² is then reproduced to the bin's own
//! residual, where linear-in-r would miss by ≈ 0.6 % on the H-like fixture (the segment from 0
//! to the first bin at r̄ ≈ 0.32 is far wider than h/2). Beyond the last entry (r_max =
//! half-extent − 3h: 3.25 Bohr at 33³, where an H tail is still ≈ 5e-4 with 3.7 % of the
//! electron outside) the last segment's log-slope is continued — the exponential tail the table
//! is built for, constant-free — instead of a step to 0, so a chord point beyond one atom's
//! r_max neither loses that tail abruptly nor reads a promolecule of 0 (review M4); a last
//! segment that does not decay (noise) is cut to 0. At or below the first entry the table is
//! the first value.
//!
//! **Electron count (a check, not physics).** `electrons()` = 4π ∫ r² ρ̄ dr by the trapezoid rule
//! over the entries.
//!
//! # Ownership
//! [`AtomReferences`] is the cache, keyed by [`ReferenceKey`] — the scene's grid, h, σ, the
//! solver's name and SCF tolerance, and Z (ruling P10 as amended by the Task 1 review: the
//! tolerance and the teacher stay in the key, so an entry's energy carries the provenance the
//! surrogate's stricter use needs; the radial table itself is insensitive to the tolerance at the
//! 1e-5 level). The driver (`crate::dynamics`) owns one and solves what the scene needs; the
//! sampling pass (`providers::sampling`) solves its own through the same [`AtomScan`] and seeds
//! the surrogate's cache from the driver's (`Trainable::start_pass`). A loaded sidecar seeds an
//! energy without a table ([`AtomReferences::seed_energy`]); [`AtomReferences::missing`] counts
//! such an entry as missing, so the table is solved when first needed.
use crate::provider::{ForceProvider, ProviderError};
use crate::sampling::allowed_multiplicities;
use crate::scene::{neumaier, Cell, IonicPolicy, Nucleus, Scene, CELL_MARGIN_H};

/// The spherically averaged density of an isolated atom as (r, ln ρ) entries in ascending r
/// (module doc, "Shell binning"); empty before the atom was solved or when the provider holds
/// no density (`sample_density` → `None`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadialDensity {
    pub r: Vec<f64>,
    pub log_rho: Vec<f64>,
}

/// Values at or below this are clamped before the log (an exact zero in a far node).
const RHO_CLAMP: f64 = 1e-300;

impl RadialDensity {
    /// Shell-bins `values` at `nodes` by |node − centre| with bins of width `dr`: each non-empty
    /// bin → (mean r, mean ln ρ), ascending r (module doc). `values` ≤ 0 are clamped to 1e-300.
    pub fn from_shells(nodes: &[[f64; 3]], values: &[f64], centre: [f64; 3], dr: f64) -> RadialDensity {
        assert_eq!(nodes.len(), values.len(), "from_shells: {} nodes, {} values", nodes.len(), values.len());
        assert!(dr > 0.0 && dr.is_finite(), "from_shells: dr = {dr}");
        // (bin, Σr, Σ ln ρ, count), bins sparse
        let mut bins: Vec<(usize, f64, f64, usize)> = Vec::new();
        for (p, &v) in nodes.iter().zip(values) {
            let d: [f64; 3] = std::array::from_fn(|a| p[a] - centre[a]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let b = (r / dr).floor() as usize;
            let l = v.max(RHO_CLAMP).ln();
            match bins.iter_mut().find(|e| e.0 == b) {
                Some(e) => {
                    e.1 += r;
                    e.2 += l;
                    e.3 += 1;
                }
                None => bins.push((b, r, l, 1)),
            }
        }
        bins.sort_unstable_by_key(|e| e.0);
        RadialDensity {
            r: bins.iter().map(|e| e.1 / e.3 as f64).collect(),
            log_rho: bins.iter().map(|e| e.2 / e.3 as f64).collect(),
        }
    }

    /// A table from an analytic profile `rho(r)` on r = 0, dr, 2dr, … ≤ r_max (tests and the
    /// analytic fixtures; the same evaluation rules as a binned table).
    pub fn from_fn(rho: impl Fn(f64) -> f64, r_max: f64, dr: f64) -> RadialDensity {
        let n = (r_max / dr).floor() as usize;
        let r: Vec<f64> = (0..=n).map(|k| k as f64 * dr).collect();
        let log_rho = r.iter().map(|&x| rho(x).max(RHO_CLAMP).ln()).collect();
        RadialDensity { r, log_rho }
    }

    pub fn is_empty(&self) -> bool {
        self.r.is_empty()
    }

    /// ρ̄(r) (module doc, "Evaluation"): log-linear between neighbouring entries (linear in r² on
    /// the segment from r = 0), the first value at or below the first entry; beyond the last the
    /// last segment's log-slope is continued (0 when that slope is not negative, or with a single
    /// entry).
    pub fn at(&self, r: f64) -> f64 {
        let n = self.r.len();
        if n == 0 || !(r >= 0.0) {
            return 0.0;
        }
        if r <= self.r[0] {
            return self.log_rho[0].exp();
        }
        if r > self.r[n - 1] {
            // the last segment's log-slope continued (module doc, "Evaluation")
            if n < 2 {
                return 0.0;
            }
            let slope = (self.log_rho[n - 1] - self.log_rho[n - 2]) / (self.r[n - 1] - self.r[n - 2]);
            return if slope < 0.0 { (self.log_rho[n - 1] + slope * (r - self.r[n - 1])).exp() } else { 0.0 };
        }
        // first index with r[i] > r: the segment is [i − 1, i]
        let i = self.r.partition_point(|&x| x <= r);
        if i == n {
            // r == r[n − 1] exactly
            return self.log_rho[n - 1].exp();
        }
        let (r0, r1) = (self.r[i - 1], self.r[i]);
        let (l0, l1) = (self.log_rho[i - 1], self.log_rho[i]);
        let t = if i == 1 && r0 == 0.0 { (r * r) / (r1 * r1) } else { (r - r0) / (r1 - r0) };
        (l0 + (l1 - l0) * t).exp()
    }

    /// 4π ∫ r² ρ̄ dr over the entries — a sanity check, not physics: the trapezoid rule between
    /// entries, except the segment from r = 0, where r²ρ̄ is far from linear (ρ̄ is flat, so the
    /// integral is ≈ ρ₀r₁³/3 and a trapezoid would give ρ₀r₁³/2: +0.0175 e on the Gaussian
    /// fixture, review M5) and Simpson's rule with the r² interpolant's midpoint is used.
    pub fn electrons(&self) -> f64 {
        let f = |k: usize| self.r[k] * self.r[k] * self.log_rho[k].exp();
        let n = self.r.len();
        if n < 2 {
            return 0.0;
        }
        let first = if self.r[0] == 0.0 {
            let r1 = self.r[1];
            let mid = 0.5 * r1;
            r1 / 6.0 * (4.0 * mid * mid * self.at(mid) + f(1))
        } else {
            0.5 * (f(0) + f(1)) * (self.r[1] - self.r[0])
        };
        4.0 * std::f64::consts::PI * (first + neumaier((2..n).map(|k| 0.5 * (f(k - 1) + f(k)) * (self.r[k] - self.r[k - 1]))))
    }
}

/// The settings an isolated atom was solved at — the atom's own provenance: the scene's grid
/// (`grid_n`, `h`), σ (the scene's, or h when it has none), the solver and its SCF tolerance
/// (0 when the solver reports none, `ForceProvider::scf_tol`); neutral, at its lowest-energy
/// multiplicity (ruling S9) — and its Z. Two keys are equal when every field is.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceKey {
    pub grid_n: usize,
    pub h: f64,
    pub sigma: f64,
    pub tol: f64,
    pub teacher: String,
    pub z: u8,
    /// The ionic surface the atom was solved on (M7; `ForceProvider::ionic_tag`, `providers::pseudo::provenance_tag`):
    /// `"none"` for the all-electron surface, else `"<policy>/<recipe hash>"`. An atom solved on one surface is never
    /// reused on another — the valence energies differ by tens of Ha.
    pub ionic: String,
}

impl ReferenceKey {
    /// The key of species `z` at `cell` and `sigma` (`None` = h), solved by `teacher` at `tol` on the ionic
    /// surface `ionic` (module doc; `"none"` = all-electron).
    pub fn new(cell: &Cell, sigma: Option<f64>, tol: Option<f64>, teacher: &str, z: u8, ionic: &str) -> ReferenceKey {
        ReferenceKey { grid_n: cell.n, h: cell.h, sigma: sigma.unwrap_or(cell.h), tol: tol.unwrap_or(0.0), teacher: teacher.to_string(), z, ionic: ionic.to_string() }
    }
}

/// One isolated-atom reference: the energy (Hartree) with the multiplicity that won the scan
/// (`None` for an energy carried by a loaded model), the tolerance and the solver it was solved
/// at (the key's, kept with the entry — spec §3.2), and the radial density (empty until solved
/// with a density, module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct AtomReference {
    pub energy: f64,
    pub multiplicity: Option<u32>,
    pub tol: f64,
    pub provider: String,
    pub radial: RadialDensity,
}

/// The references solved so far (module doc, "Ownership").
#[derive(Clone, Debug, Default)]
pub struct AtomReferences {
    entries: Vec<(ReferenceKey, AtomReference)>,
}

impl AtomReferences {
    /// The reference stored under `key`, if any (a seeded energy without a table included).
    pub fn get(&self, key: &ReferenceKey) -> Option<&AtomReference> {
        self.entries.iter().find(|(k, _)| k == key).map(|e| &e.1)
    }

    /// Stores `r` under `key`, replacing an entry with the same key.
    pub fn insert(&mut self, key: ReferenceKey, r: AtomReference) {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = r,
            None => self.entries.push((key, r)),
        }
    }

    /// Seeds an energy without a table (a loaded sidecar): inserted unless the key already holds
    /// an entry with a table. [`missing`](Self::missing) still reports it, so the table is solved
    /// when first needed and the entry then replaced.
    pub fn seed_energy(&mut self, key: ReferenceKey, e: f64, multiplicity: Option<u32>, tol: f64, provider: &str) {
        if self.get(&key).is_some_and(|r| !r.radial.is_empty()) {
            return;
        }
        self.insert(key, AtomReference { energy: e, multiplicity, tol, provider: provider.to_string(), radial: RadialDensity::default() });
    }

    /// The species of `species` (in order, each once) with no entry under `key_for(z)` or an entry
    /// without a radial table.
    pub fn missing(&self, key_for: impl Fn(u8) -> ReferenceKey, species: &[u8]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for &z in species {
            if out.contains(&z) {
                continue;
            }
            if self.get(&key_for(z)).is_none_or(|r| r.radial.is_empty()) {
                out.push(z);
            }
        }
        out
    }

    /// Every entry, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &(ReferenceKey, AtomReference)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The radius the shell nodes of `cell` reach: half-extent − 3h (`CELL_MARGIN_H`), the margin the
/// providers keep every nucleus inside, so the table never reads the Dirichlet wall's cut-off.
pub fn shell_radius(cell: &Cell) -> f64 {
    cell.half_extent() - CELL_MARGIN_H * cell.h
}

/// The grid nodes of `cell` within `r_max` of its centre (the origin: `qm::Grid::cube` is centred
/// there — the atom's position), in `qm::Grid` order (x fastest).
pub fn shell_nodes(cell: &Cell, r_max: f64) -> Vec<[f64; 3]> {
    let half = cell.half_extent();
    let coord = |i: usize| -half + i as f64 * cell.h;
    let mut out = Vec::new();
    for k in 0..cell.n {
        for j in 0..cell.n {
            for i in 0..cell.n {
                let p = [coord(i), coord(j), coord(k)];
                if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= r_max * r_max {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// The multiplicity scan of one species, one solve per [`step`](Self::step) (module doc, "The
/// scan"): the sampling pass advances it one teacher solve per tick and resumes it after a
/// cancelled solve; [`solve_atom`] runs it to the end. Each step is cold (`reset()` before it)
/// and reads the density at the shell nodes afterwards (one `sample_density` of ≤ n³ points).
#[derive(Clone, Debug)]
pub struct AtomScan {
    z: u8,
    a: u16,
    mass: f64,
    cell: Cell,
    sigma: Option<f64>,
    ionic: IonicPolicy,
    remaining: Vec<u32>,
    /// (multiplicity, energy, its radial density), in the order solved.
    tried: Vec<(u32, f64, RadialDensity)>,
    failed: Vec<(u32, String)>,
}

impl AtomScan {
    /// The scan of species `z` (mass number `a`, mass in mₑ) on `cell` at `sigma` (`None` = h) under the ionic
    /// policy `ionic` (the scene's: a lone O under `CoresOnly` is the pseudo-O the molecule's bonds are read against).
    pub fn new(cell: &Cell, sigma: Option<f64>, ionic: IonicPolicy, z: u8, a: u16, mass: f64) -> AtomScan {
        AtomScan { z, a, mass, cell: cell.clone(), sigma, ionic, remaining: allowed_multiplicities(z as u32), tried: Vec::new(), failed: Vec::new() }
    }

    pub fn z(&self) -> u8 {
        self.z
    }

    /// Multiplicities still to solve, in order.
    pub fn remaining(&self) -> &[u32] {
        &self.remaining
    }

    /// (multiplicity, energy) of the solves that succeeded, in order.
    pub fn tried(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.tried.iter().map(|t| (t.0, t.1))
    }

    /// (multiplicity, error) of the solves that failed.
    pub fn failed(&self) -> &[(u32, String)] {
        &self.failed
    }

    /// A solve has been attempted and not cancelled.
    pub fn started(&self) -> bool {
        !self.tried.is_empty() || !self.failed.is_empty()
    }

    pub fn is_done(&self) -> bool {
        self.remaining.is_empty()
    }

    /// The isolated neutral atom at the cell centre with multiplicity `m`: the scan's cell and σ.
    pub fn scene(&self, m: u32) -> Scene {
        Scene {
            nuclei: vec![Nucleus { pos: [0.0; 3], vel: [0.0; 3], z: self.z, a: self.a, mass: self.mass }],
            cell: self.cell.clone(),
            charge: 0,
            multiplicity: Some(m),
            sigma: self.sigma,
            external_field: [0.0; 3],
            ionic: self.ionic,
        }
    }

    /// The next multiplicity, solved cold: `Ok(Some((m, E)))` after it (its density read and
    /// binned); `Ok(None)` when nothing is left; `Err(Cancelled)` leaves the multiplicity queued;
    /// any other error (the solver's, or an answer with the wrong shape or a non-finite number)
    /// drops that multiplicity — recorded in [`failed`](Self::failed) — and is returned.
    pub fn step(&mut self, provider: &mut dyn ForceProvider, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<(u32, f64)>, ProviderError> {
        let Some(&m) = self.remaining.first() else { return Ok(None) };
        let scene = self.scene(m);
        provider.reset();
        let outcome = provider.evaluate(&scene, should_abort).and_then(|r| {
            if r.forces.len() != 1 {
                return Err(ProviderError::Invalid(format!("provider '{}' returned {} forces for 1 nucleus", provider.name(), r.forces.len())));
            }
            if !r.energy.is_finite() || r.forces.iter().flatten().any(|x| !x.is_finite()) {
                return Err(ProviderError::Invalid(format!("provider '{}' returned a non-finite energy or force", provider.name())));
            }
            Ok(r.energy)
        });
        match outcome {
            Err(ProviderError::Cancelled) => Err(ProviderError::Cancelled),
            Err(e) => {
                self.remaining.remove(0);
                self.failed.push((m, e.to_string()));
                Err(e)
            }
            Ok(e) => {
                self.remaining.remove(0);
                let radial = read_radial(provider, &self.cell);
                self.tried.push((m, e, radial));
                Ok(Some((m, e)))
            }
        }
    }

    /// The entry: the lowest energy of the solved multiplicities (ruling S9) with its density;
    /// `None` when nothing was solved. `tol` and `provider` are recorded with it.
    pub fn finish(self, tol: f64, provider: &str) -> Option<AtomReference> {
        let best = self.tried.into_iter().min_by(|a, b| a.1.total_cmp(&b.1))?;
        Some(AtomReference { energy: best.1, multiplicity: Some(best.0), tol, provider: provider.to_string(), radial: best.2 })
    }
}

/// The provider's current density at the shell nodes of `cell`, binned with dr = h/2; empty when
/// it holds none.
pub fn read_radial(provider: &mut dyn ForceProvider, cell: &Cell) -> RadialDensity {
    let nodes = shell_nodes(cell, shell_radius(cell));
    match provider.sample_density(&nodes) {
        Some(v) if v.len() == nodes.len() => RadialDensity::from_shells(&nodes, &v, [0.0; 3], 0.5 * cell.h),
        _ => RadialDensity::default(),
    }
}

/// One isolated neutral atom of species `z` at the centre of `cell` at `sigma` (`None` = h):
/// every allowed multiplicity solved cold, the lowest energy kept with its shell-binned density
/// (module doc). `provider` is reset before every solve and once after the scan, so its warm
/// state is never the atom's; `tol` and the name are recorded from it. `Err(Cancelled)` when
/// `should_abort` ended a solve (the provider is reset); a multiplicity whose solve fails is left
/// out, and the error is returned only when none succeeded.
pub fn solve_atom(provider: &mut dyn ForceProvider, cell: &Cell, sigma: Option<f64>, ionic: IonicPolicy, z: u8, mass: f64, should_abort: &mut dyn FnMut() -> bool) -> Result<AtomReference, ProviderError> {
    solve_atom_reporting(provider, cell, sigma, ionic, z, mass, should_abort, &mut |_| {})
}

/// The multiplicities [`solve_atom`] solves for species `z` (the scan's length: the "of n" of a
/// progress report).
pub fn multiplicities(z: u8) -> usize {
    allowed_multiplicities(z as u32).len()
}

/// [`solve_atom`], calling `done(k)` after the k-th multiplicity's solve (solved or failed) —
/// the progress the driver shows for a reference solving off the stepping path.
pub fn solve_atom_reporting(provider: &mut dyn ForceProvider, cell: &Cell, sigma: Option<f64>, ionic: IonicPolicy, z: u8, mass: f64, should_abort: &mut dyn FnMut() -> bool, done: &mut dyn FnMut(usize)) -> Result<AtomReference, ProviderError> {
    let a = units::most_abundant_a(z).unwrap_or(0);
    let mut scan = AtomScan::new(cell, sigma, ionic, z, a, mass);
    let total = scan.remaining().len();
    let mut last_err: Option<ProviderError> = None;
    while !scan.is_done() {
        let r = scan.step(provider, should_abort);
        if !matches!(r, Err(ProviderError::Cancelled)) {
            done(total - scan.remaining().len());
        }
        match r {
            Ok(_) => {}
            Err(ProviderError::Cancelled) => {
                provider.reset();
                return Err(ProviderError::Cancelled);
            }
            Err(e) => last_err = Some(e),
        }
    }
    provider.reset();
    let (tol, name) = (provider.scf_tol(), provider.name());
    scan.finish(tol.unwrap_or(0.0), name).ok_or_else(|| last_err.unwrap_or_else(|| ProviderError::Invalid(format!("no multiplicity of Z = {z} could be solved"))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{EvalStats, ForceResult};
    use crate::test_providers::Gaussians;
    use std::f64::consts::PI;

    /// A Gaussian atom (A = 0.3, w = 0.8) at the centre of a 49³ h 0.25 cell through
    /// `solve_atom`: the binned table reproduces A·exp(−r²/2w²) at r ∈ {0, 0.3, 0.7, 1.4, 2.9}
    /// within the scheme's own bound; the first entry is r = 0 with the centre value; no entry in
    /// (0, 0.25) (empty bins are absent); `electrons()` = A(2π)^{3/2}w³ to 1e-2.
    ///
    /// The bound, derived (module doc, "Evaluation"): log-linear interpolation of ln ρ = −r²/2w²
    /// between entries Δ apart errs by ≤ (Δ²/8)·|(ln ρ)″| = Δ²/(8w²); the entries are bin
    /// *means*, so Δ is not dr = h/2 but up to Δ_max read off the table (0.142 = 1.14 dr beyond
    /// r = 1 here; review M2); plus the bin's own residual Var_b(r)/(2w²) ≤ (dr²/4)/(2w²) at
    /// worst (two radii at the bin's edges) — (Δ_max² + dr²)/(8w²) = 7.0e-3 together. The
    /// brief's 2e-3 does not follow from the scheme it specifies (the interpolation term alone
    /// is 3.7e-3 at r = 1.4) and is replaced by the derived bound. Measured max 4.4e-3 (at
    /// r = 1.4) over the five points, 5.0e-3 over a sweep of r ∈ [0.3, 5] — printed.
    #[test]
    fn shell_binning_a_gaussian_atom_reproduces_its_profile_log_linearly() {
        let (a, w): (f64, f64) = (0.3, 0.8);
        let dr = 0.125;
        let mut g = Gaussians::per_atom(a, w);
        let cell = Cell { n: 49, h: 0.25 };
        let r = solve_atom(&mut g, &cell, None, IonicPolicy::None, 1, 1837.15, &mut || false).unwrap();
        assert_eq!(r.multiplicity, Some(2));
        assert_eq!((r.tol, r.provider.as_str()), (0.0, "gaussians"));
        let t = &r.radial;
        // the log-linear segments (the first, from r = 0, is linear in r² and not in this bound)
        let d_max = t.r[1..].windows(2).map(|p| p[1] - p[0]).fold(0.0f64, f64::max);
        let bound = (d_max * d_max + dr * dr) / (8.0 * w * w);
        println!("entry spacing: max {d_max:.4} (dr = {dr}); bound (Δ_max² + dr²)/(8w²) = {bound:.2e}");
        assert!(!t.is_empty());
        assert_eq!(t.r[0], 0.0);
        assert!((t.log_rho[0].exp() - a).abs() <= 1e-12, "centre value {}", t.log_rho[0].exp());
        assert!(t.r[1] >= 0.25, "first bin after the centre at r = {}", t.r[1]);
        let exact = |x: f64| a * (-x * x / (2.0 * w * w)).exp();
        let mut worst = 0.0f64;
        for x in [0.0, 0.3, 0.7, 1.4, 2.9] {
            let rel = (t.at(x) - exact(x)).abs() / exact(x);
            worst = worst.max(rel);
            println!("r = {x}: table {:.6e} exact {:.6e} rel {rel:.2e} (bound {bound:.2e})", t.at(x), exact(x));
            assert!(rel <= bound, "r = {x}: {rel:.2e} > the derived bound {bound:.2e}");
        }
        let n_e = t.electrons();
        let n_exact = a * (2.0 * PI).powf(1.5) * w * w * w;
        println!("electrons: {n_e:.5} vs {n_exact:.5} ({:.2e} relative); worst profile error {worst:.2e}; {} entries, last r {:.3}", (n_e / n_exact - 1.0).abs(), t.r.len(), t.r.last().unwrap());
        assert!((n_e / n_exact - 1.0).abs() <= 1e-2);
        // the sweep: the worst relative error over r ∈ [0.3, 5]
        let sweep = (0..=470).map(|k| 0.3 + 0.01 * k as f64).map(|x| (t.at(x) - exact(x)).abs() / exact(x)).fold(0.0f64, f64::max);
        println!("worst relative error over r ∈ [0.3, 5] at 471 points: {sweep:.2e}");
        assert!(sweep <= bound, "sweep {sweep:.2e} > bound {bound:.2e}");
        // beyond the last entry the last log-slope continues (no step to 0); r < 0 is 0
        let last = *t.r.last().unwrap();
        assert!((t.at(last + 1e-9) / t.at(last) - 1.0).abs() < 1e-6, "continuous at the last entry");
        assert!(t.at(last + 0.5) < t.at(last) && t.at(last + 0.5) > 0.0);
        assert!(t.at(100.0) < 1e-30 && t.at(-1.0) == 0.0);
        // r_max: half-extent − 3h = 6 − 0.75
        assert!(*t.r.last().unwrap() <= shell_radius(&cell) && *t.r.last().unwrap() > shell_radius(&cell) - 0.25);
    }

    /// A scripted provider whose energy depends on the multiplicity (m = 1 → −1.0, m = 3 →
    /// −1.2; Z = 2): the triplet wins; every multiplicity cold (a reset before each, one after:
    /// 3 for helium — the brief's "twice" is a one-multiplicity species); tol and the name are
    /// recorded; the table is empty (an analytic provider without a density).
    #[test]
    fn solve_atom_keeps_the_lowest_multiplicity_and_resets_the_provider() {
        struct ByMultiplicity {
            resets: usize,
            seen: Vec<(u32, usize)>,
        }
        impl ForceProvider for ByMultiplicity {
            fn evaluate(&mut self, scene: &Scene, _a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
                let m = scene.multiplicity.unwrap();
                self.seen.push((m, self.resets));
                let energy = match m { 1 => -1.0, 3 => -1.2, _ => panic!("multiplicity {m}") };
                Ok(ForceResult { energy, forces: vec![[0.0; 3]], stats: EvalStats::default(), display: None, orbitals: None })
            }
            fn name(&self) -> &'static str {
                "scripted"
            }
            fn reset(&mut self) {
                self.resets += 1;
            }
            fn scf_tol(&self) -> Option<f64> {
                Some(1e-7)
            }
        }
        let mut p = ByMultiplicity { resets: 0, seen: Vec::new() };
        let cell = Cell { n: 17, h: 0.5 };
        let r = solve_atom(&mut p, &cell, Some(0.3), IonicPolicy::None, 2, 7000.0, &mut || false).unwrap();
        assert_eq!((r.multiplicity, r.energy), (Some(3), -1.2));
        assert_eq!((r.tol, r.provider.as_str()), (1e-7, "scripted"));
        assert!(r.radial.is_empty());
        assert_eq!(p.seen, vec![(1, 1), (3, 2)], "each multiplicity after its own reset");
        assert_eq!(p.resets, 3, "reset before each of the two multiplicities and once after");
        // the scan is resumable: a cancelled solve leaves the multiplicity queued
        struct CancelOnce(bool);
        impl ForceProvider for CancelOnce {
            fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
                if a() {
                    return Err(ProviderError::Cancelled);
                }
                Ok(ForceResult { energy: -(scene.multiplicity.unwrap() as f64), forces: vec![[0.0; 3]], stats: EvalStats::default(), display: None, orbitals: None })
            }
            fn name(&self) -> &'static str {
                "cancel once"
            }
            fn reset(&mut self) {
                self.0 = true;
            }
        }
        let mut c = CancelOnce(false);
        let mut scan = AtomScan::new(&cell, None, IonicPolicy::None, 2, 4, 7000.0);
        assert_eq!(scan.step(&mut c, &mut || false).unwrap(), Some((1, -1.0)));
        assert!(matches!(scan.step(&mut c, &mut || true), Err(ProviderError::Cancelled)));
        assert_eq!(scan.remaining(), &[3]);
        assert_eq!(scan.step(&mut c, &mut || false).unwrap(), Some((3, -3.0)));
        assert!(scan.is_done() && scan.step(&mut c, &mut || false).unwrap().is_none());
        assert_eq!(scan.finish(0.0, "x").unwrap().multiplicity, Some(3));
    }

    #[test]
    fn missing_counts_seeded_energies_without_a_table() {
        let cell = Cell { n: 33, h: 0.25 };
        let key = |z: u8| ReferenceKey::new(&cell, None, Some(1e-6), "t", z, "none");
        let mut refs = AtomReferences::default();
        assert_eq!(refs.missing(key, &[1, 8, 1]), vec![1, 8]);
        refs.seed_energy(key(1), -0.478, None, 1e-6, "t");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs.get(&key(1)).map(|r| (r.energy, r.multiplicity)), Some((-0.478, None)));
        assert_eq!(refs.missing(key, &[1, 8]), vec![1, 8], "a seeded energy has no table");
        let full = AtomReference { energy: -0.47, multiplicity: Some(2), tol: 1e-6, provider: "t".into(), radial: RadialDensity::from_fn(|r| (-2.0 * r).exp(), 5.0, 0.125) };
        refs.insert(key(1), full.clone());
        assert_eq!(refs.missing(key, &[1, 8]), vec![8]);
        // a seed never overwrites a solved table
        refs.seed_energy(key(1), -0.5, None, 1e-6, "t");
        assert_eq!(refs.get(&key(1)), Some(&full));
        // another tolerance or solver is another key
        assert_eq!(refs.missing(|z| ReferenceKey::new(&cell, None, Some(1e-5), "t", z, "none"), &[1]), vec![1]);
        assert_eq!(refs.missing(|z| ReferenceKey::new(&cell, None, Some(1e-6), "u", z, "none"), &[1]), vec![1]);
        assert_eq!(ReferenceKey::new(&cell, None, None, "t", 1, "none"), ReferenceKey { grid_n: 33, h: 0.25, sigma: 0.25, tol: 0.0, teacher: "t".into(), z: 1, ionic: "none".into() });
    }

    /// The interpolation rules: exact for an exponential tail beyond the first segment, exact for
    /// a Gaussian on the first segment (linear in r² from r = 0); `from_shells` bins by distance
    /// from the given centre.
    #[test]
    fn log_linear_is_exact_for_an_exponential_and_bins_by_distance_from_the_centre() {
        let t = RadialDensity::from_fn(|r| 0.7 * (-2.3 * r).exp(), 4.0, 0.125);
        for r in [0.2f64, 0.33, 1.0, 2.71, 3.9] {
            let exact = 0.7 * (-2.3 * r).exp();
            assert!((t.at(r) - exact).abs() <= 1e-12 * exact, "r = {r}: {} vs {exact}", t.at(r));
        }
        // beyond the last entry the last log-slope continues: exact for the exponential
        let beyond = 0.7 * (-2.3 * 4.5f64).exp();
        assert!((t.at(4.5) - beyond).abs() <= 1e-12 * beyond, "{} vs {beyond}", t.at(4.5));
        assert_eq!(t.at(0.0), 0.7);
        // the first segment is linear in r²: a Gaussian is reproduced there, an exponential is
        // not (its cusp has no even expansion) — the smeared atom is the former
        let gauss = RadialDensity::from_fn(|r| 0.4 * (-r * r / 1.3).exp(), 4.0, 0.125);
        for r in [0.03f64, 0.0625, 0.11] {
            let exact = 0.4 * (-r * r / 1.3).exp();
            assert!((gauss.at(r) - exact).abs() <= 1e-12 * exact, "r = {r}: {} vs {exact}", gauss.at(r));
        }
        assert!((t.at(0.0625) / (0.7 * (-2.3 * 0.0625f64).exp()) - 1.0).abs() > 1e-3, "an exponential's first segment is not even in r");
        // nodes of a small cell around an off-centre point: the entries are the distinct shells
        let cell = Cell { n: 5, h: 0.5 };
        let nodes = shell_nodes(&cell, 10.0);
        assert_eq!(nodes.len(), 125);
        let centre = [0.5, 0.0, -0.5];
        let values: Vec<f64> = nodes.iter().map(|p| {
            let d: [f64; 3] = std::array::from_fn(|a| p[a] - centre[a]);
            (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).exp()
        }).collect();
        // dr = 0.1 separates the shells at 0.5 and 0.707 (dr = 0.25 would bin them together)
        let b = RadialDensity::from_shells(&nodes, &values, centre, 0.1);
        assert_eq!(b.r[0], 0.0);
        assert_eq!(b.log_rho[0], 0.0);
        assert!((b.r[1] - 0.5).abs() < 1e-12 && (b.log_rho[1] + 0.25).abs() < 1e-12, "{:?}", (&b.r[..3], &b.log_rho[..3]));
        assert!(b.r.windows(2).all(|w| w[0] < w[1]), "ascending");
        // a zero value is clamped, not a NaN
        let z = RadialDensity::from_shells(&[[0.0; 3], [1.0, 0.0, 0.0]], &[1.0, 0.0], [0.0; 3], 0.5);
        assert!(z.log_rho[1].is_finite() && z.at(1.0) > 0.0 && z.at(1.0) < 1e-200);
    }
}

#[cfg(test)]
mod ionic_tests {
    use super::*;

    /// M7 Task 9: an atom solved on one ionic surface is never reused on another (the valence energies differ
    /// by tens of Ha): the surface is part of the key.
    #[test]
    fn a_reference_is_keyed_by_its_ionic_surface() {
        let cell = Cell { n: 33, h: 0.25 };
        let a = ReferenceKey::new(&cell, None, Some(1e-6), "t", 8, "none");
        let b = ReferenceKey::new(&cell, None, Some(1e-6), "t", 8, "cores/0123456789abcdef/shared1e-3");
        assert_ne!(a, b);
        let mut refs = AtomReferences::default();
        refs.seed_energy(a.clone(), -41.5, None, 1e-6, "t");
        assert!(refs.get(&a).is_some() && refs.get(&b).is_none(), "the pseudo surface does not see the all-electron atom");
    }

    /// The atom scan solves the lone atom under the scene's policy.
    #[test]
    fn the_scan_scene_carries_the_policy() {
        let cell = Cell { n: 33, h: 0.25 };
        for p in IonicPolicy::ALL {
            let scan = AtomScan::new(&cell, None, p, 8, 16, 29166.0);
            assert_eq!(scan.scene(3).ionic, p);
        }
    }
}
