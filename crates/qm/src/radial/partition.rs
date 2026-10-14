//! The core/valence partition by a universal rule (spec §4.2; review S5, S9). CPU, f64, atomic units.
//! Input: the spin-restricted reference atom of §4.1 (spec §4.3: "Reference configuration: the
//! spin-restricted ground configuration"). The only per-element input is Z (through that atom); the
//! window Δ_core and l_max's offset are `LEDGER.delta_core` and `LEDGER.l_max_above_valence`.
//!
//! # The rule (spec §4.2, verbatim)
//! 1. Seed: every occupied shell with the highest occupied principal quantum number n_max is valence.
//! 2. Promote whole lower shells: the shell n = n_max − 1 (all its subshells) is promoted to valence —
//!    "semicore" — if any of its occupied subshells has ε_nl > ε_v,min − Δ_core, with ε_v,min the
//!    lowest seed eigenvalue and **Δ_core = 2.0 Ha** (universal). Repeat for n_max − 2 (never triggers
//!    below the f block). Everything else is core.
//! 3. Channels: a pseudopotential channel is generated for every l ≤ l_max, l_max = max(l occupied in
//!    the valence) + 1, provided a bound state of that l exists in the spin-restricted reference
//!    potential. An unbound channel is dropped and recorded in `dropped`; its scattering is left to the
//!    local part (§4.4). For an unoccupied channel, r_c,l is **capped at max_{l occupied} r_c,l** (review
//!    S5), so a diffuse, barely bound state cannot set a projector sphere larger than the occupied
//!    channels'. Where the cap binds (measured, Task 2): Be, whose unoccupied 2p (ε = −0.077 Ha, r_max
//!    2.28 Bohr) takes r_c,s at every f. Its r_c can then lie inside its own r_max (2.04 < 2.28 at
//!    f = 1.0), which is harmless for a nodeless state (§4.4). The neutral LSDA atom's potential has no
//!    −1/r tail (self-interaction), so it has no Rydberg series. The 3d is unbound for B–Ne *and* for
//!    Al–Ar (measured, Task 2). The earlier "3d of Al–Ar at ε ≈ −0.02 Ha" was the −1/r-tail estimate and
//!    does not occur. B–Ne and Al–Ar therefore have s and p channels only: one s projector, p local
//!    (§4.4 item 1). For Al–Ar this means the d-wave scattering is the p-local potential's. It is not
//!    gated (T5 excludes an ungenerated l). G2's count does check that this potential binds no spurious
//!    d state, and the d log-derivative mismatch is reported (Task 12). It is a stated limitation until
//!    the ONCV stage, which can build an unbound channel from a scattering state.
//!    Scope note for Z > 18 (review S9, not this milestone): "promote the whole n_max − 1 shell" would
//!    promote 3s, 3p and 3d together for Sc–Zn and Ga–Kr (3d within 2 Ha of 4s) and refuse the whole d
//!    block through a 3s/4s two-s-state refusal; the subshell-wise window (promote only the subshells
//!    inside Δ_core) is equally universal and is what practice does (3d semicore only). The rule is
//!    revisited with the ONCV stage before the d block is enabled.
//! 4. Consequences (derived, not chosen): H and He have no core → no pseudopotential under the
//!    default policy (§4.7 — they stay all-electron smeared, identical to today); under `All` the
//!    no-core branch is explicit in the partition/NLCC code (ρ_c ≡ 0, N_c = 0, r_nlc undefined,
//!    ρ_pc = 0, r_core = 0 in §5.6). Li: 1s (−1.9 Ha) is within 2 Ha of 2s (−0.1) → the whole n = 1
//!    shell is valence → the s channel carries two valence states, which a one-projector TM/KB
//!    potential cannot represent (§4.3) → the generator **refuses** Li (reported, not patched) and Li
//!    remains the all-electron known limitation until ONCV. Be–Ne: 1s core (Be 1s ≈ −3.9 Ha vs 2s
//!    −0.2: gap 3.7). Na, Mg: the n = 2 shell promoted (Na 2p ≈ −1.0 Ha vs 3s −0.1) → 9/10-electron
//!    semicore potentials, which again need two s-projectors → ONCV stage. Al–Ar: n = 2 core (Al 2p ≈
//!    −2.6 Ha vs 3p −0.1). These eigenvalues are quoted here only to show what the rule does; the code
//!    reads them from its own atom.
//!
//! # Conventions fixed here (plan Task 2; used by every later task)
//! - Occupations are both spins of the restricted reference (`Shell::occ`, `Channel::occ`).
//! - Channels are ascending in l; the "highest generated channel" is `channels.last()`.
//! - The unoccupied l_max channel is the lowest bound state of that l in the reference potential,
//!   `next_unoccupied(reference, l, 0)` (its node count follows from the core shells of that l); if
//!   unbound it is dropped and listed in `dropped`. Its r_c cap is applied in `tm::tm_channels`.
//! - No-core branch (item 4): `core` empty, `n_core = 0`, `z_v = Z`, `core_density` ≡ 0.
//!
//! Densities: ρ(r) = Σ_shells f_nl u_nl(r)² / (4πr²) (e/Bohr³), u = rR normalised ∫u² dr = 1.
use super::atom::{ground_state, next_unoccupied, AtomResult, SpinMode};
use super::ledger::LEDGER;
use super::numerov::bound_state;
use std::f64::consts::PI;

/// An occupied shell of the restricted reference; `occ` counts both spins.
#[derive(Clone, Debug, PartialEq)]
pub struct Shell { pub n: u8, pub l: u8, pub eps: f64, pub occ: f64 }

/// One pseudopotential channel: the reference state of angular momentum l (the valence shell of that
/// l, or the lowest unoccupied bound state for an unoccupied l ≤ l_max), u positive at r_peak.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel { pub l: u8, pub n: u8, pub eps: f64, pub occ: f64, pub r_peak: f64, pub u: Vec<f64>, pub occupied: bool }

#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    pub z: u8, pub core: Vec<Shell>, pub valence: Vec<Shell>, pub n_core: u32, pub z_v: f64,
    pub semicore: bool,
    /// The rule's l_max (max valence l + 1), before unbound channels are dropped. The highest *generated*
    /// channel is `channels.last()` and can be lower (H/He: 1 vs 0; B–Ne and Al–Ar: 2 vs 1): take the
    /// local channel from `channels.last()`, and iterate 0..=l_max only where a dropped l must count zero
    /// states (G2) — review S2.
    pub l_max: u8,
    /// Ascending in l.
    pub channels: Vec<Channel>,
    /// The l ≤ l_max with no bound reference state (left to the local part).
    pub dropped: Vec<u8>,
    /// For each l = 0..=l_max, every bound eigenvalue (< 0, ascending) of the reference potential with
    /// angular momentum l *above that l's core shells*: the valence/channel state and any higher bound
    /// state. The neutral LSDA atom has no −1/r tail and so no Rydberg series, but its short-range well
    /// can still bind one (n+1)s level just below 0: measured C 3s −0.0061, O 3s −0.0045, Ne 3s −0.0024,
    /// Si 4s −0.0138 Ha (Task 3). G2's "no extra bound state" counts against this all-electron spectrum,
    /// not against the generated channels alone (kb module doc; Task 3 report).
    pub spectrum: Vec<Vec<f64>>,
}

pub(crate) fn l_name(l: u8) -> char { ['s', 'p', 'd', 'f', 'g', 'h', 'i'].get(l as usize).copied().unwrap_or('?') }

fn fmt_states(z: &[(u8, f64)], l: u8) -> String {
    z.iter().map(|(n, e)| format!("{n}{} at {e:.3} Ha", l_name(l))).collect::<Vec<_>>().join(", ")
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Refusal {
    #[error("Z = {z}: two valence {} states ({}) — a one-projector TM/KB potential cannot carry both; refused until the ONCV stage (spec §4.3)", l_name(*l), fmt_states(states, *l))]
    TwoStatesSameL { z: u8, l: u8, states: Vec<(u8, f64)> },
    #[error("Z = {z}: no entry of the f ladder passes the gates at this recipe version: {}", tried.join("; "))]
    NoLadderEntryPasses { z: u8, tried: Vec<String> },
    #[error("Z = {z}: the radial reference atom failed: {message}")]
    Radial { z: u8, message: String },
    /// The LFC fit has no solution (ρ_c not falling at r_nlc: κ ≥ 1; never met for Z ≤ 18) — Task 4 review N2.
    #[error("Z = {z}: no LFC partial core: {message}")]
    PartialCore { z: u8, message: String },
}

/// The §4.2 rule on the spin-restricted reference. Refuses a channel with two valence states.
pub fn partition(reference: &AtomResult) -> Result<Partition, Refusal> {
    assert_eq!(reference.spin, SpinMode::Restricted, "the partition is taken on the spin-restricted reference (spec §4.3)");
    let z = reference.z;
    let occupied: Vec<Shell> = reference.orbitals.iter().filter(|o| o.occ[0] + o.occ[1] > 0.0)
        .map(|o| Shell { n: o.n, l: o.l, eps: o.eps[0], occ: o.occ[0] + o.occ[1] }).collect();
    let n_max = occupied.iter().map(|s| s.n).max().ok_or_else(|| Refusal::Radial { z, message: "the reference has no electrons".into() })?;
    // 1. seed: every occupied shell with the highest n
    let mut valence: Vec<Shell> = occupied.iter().filter(|s| s.n == n_max).cloned().collect();
    let eps_v_min = valence.iter().map(|s| s.eps).fold(f64::INFINITY, f64::min);
    // 2. promote whole lower shells while any of their subshells lies within Δ_core of the seed
    let mut semicore = false;
    for n in (1..n_max).rev() {
        let shell: Vec<Shell> = occupied.iter().filter(|s| s.n == n).cloned().collect();
        if shell.iter().any(|s| s.eps > eps_v_min - LEDGER.delta_core) {
            valence.extend(shell);
            semicore = true;
        } else {
            break;
        }
    }
    let core: Vec<Shell> = occupied.iter().filter(|s| !valence.iter().any(|v| v.n == s.n && v.l == s.l)).cloned().collect();
    let n_core = core.iter().map(|s| s.occ).sum::<f64>().round() as u32;
    let z_v = valence.iter().map(|s| s.occ).sum::<f64>();
    // 3. channels l = 0..=l_max
    let l_max = valence.iter().map(|s| s.l).max().expect("valence is not empty") + LEDGER.l_max_above_valence;
    let (mut channels, mut dropped) = (Vec::new(), Vec::new());
    for l in 0..=l_max {
        let of_l: Vec<&Shell> = valence.iter().filter(|s| s.l == l).collect();
        match of_l.len() {
            0 => match next_unoccupied(reference, l, 0) {
                Some((n, b)) => channels.push(Channel { l, n, eps: b.e, occ: 0.0, r_peak: b.r_peak, u: b.u, occupied: false }),
                None => dropped.push(l),
            },
            1 => {
                let s = of_l[0];
                let o = reference.orbitals.iter().find(|o| o.n == s.n && o.l == l).expect("listed");
                channels.push(Channel { l, n: s.n, eps: s.eps, occ: s.occ, r_peak: o.r_peak, u: o.u[0].clone(), occupied: true });
            }
            _ => {
                let mut states: Vec<(u8, f64)> = of_l.iter().map(|s| (s.n, s.eps)).collect();
                states.sort_by_key(|s| s.0);
                return Err(Refusal::TwoStatesSameL { z, l, states });
            }
        }
    }
    // the reference's bound spectrum above the core, per l (node counts from the number of core shells of l)
    let spectrum = (0..=l_max).map(|l| {
        let n_core_l = core.iter().filter(|s| s.l == l).count();
        (n_core_l..).map_while(|k| bound_state(&reference.grid, &reference.v[0], l, k, z as f64, None)).map(|b| b.e).collect()
    }).collect();
    Ok(Partition { z, core, valence, n_core, z_v, semicore, l_max, channels, dropped, spectrum })
}

fn shell_density(reference: &AtomResult, shells: &[Shell]) -> Vec<f64> {
    let g = &reference.grid;
    let orbs: Vec<(f64, &[f64])> = shells.iter().map(|s| {
        let o = reference.orbitals.iter().find(|o| o.n == s.n && o.l == s.l).expect("listed");
        (s.occ, o.u[0].as_slice())
    }).collect();
    (0..g.len()).map(|i| {
        let r = g.r[i];
        orbs.iter().map(|(f, u)| f * u[i] * u[i]).sum::<f64>() / (4.0 * PI * r * r)
    }).collect()
}

/// ρ_c(r) (e/Bohr³); identically 0 without a core (the explicit no-core branch).
pub fn core_density(reference: &AtomResult, part: &Partition) -> Vec<f64> { shell_density(reference, &part.core) }

/// The all-electron valence density ρ_v^AE(r) of the reference (e/Bohr³).
pub fn valence_density(reference: &AtomResult, part: &Partition) -> Vec<f64> { shell_density(reference, &part.valence) }

/// Whether §4.2 gives Z a core (`IonicPolicy::CoresOnly`, spec §4.7). A refusal propagates
/// (the caller falls back to all-electron under `CoresOnly`).
pub fn has_core(z: u8) -> Result<bool, Refusal> {
    let a = ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?;
    Ok(partition(&a)?.n_core > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::{ground_state, SpinMode};

    fn part(z: u8) -> Result<Partition, Refusal> { partition(&ground_state(z, SpinMode::Restricted).expect("converges")) }

    /// Spec §8 row 2: Li's 1s (−1.9 Ha) is within Δ_core of 2s → two valence s states → refused, the
    /// message naming the ONCV stage (spec §4.2 item 4, §4.3).
    #[test]
    fn lithium_is_refused_two_s_states() {
        match part(3) {
            Err(r @ Refusal::TwoStatesSameL { z: 3, l: 0, .. }) => {
                let m = r.to_string();
                eprintln!("Li: {m}");
                assert!(m.contains("two valence s states") && m.contains("ONCV"), "{m}");
            }
            other => panic!("Li must be refused with two s states, got {other:?}"),
        }
    }

    /// Be–Ne: the 1s is the core, Z_v = Z − 2, channels s and p only (the 3d is unbound in the neutral
    /// LSDA atom — spec §4.2 item 3); derived, nothing tabulated.
    #[test]
    fn first_row_cores_are_the_1s() {
        for z in 4u8..=10 {
            let p = part(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
            report(&p);
            assert_eq!(p.core.iter().map(|s| (s.n, s.l)).collect::<Vec<_>>(), vec![(1, 0)], "Z = {z}");
            assert_eq!(p.z_v, z as f64 - 2.0);
            assert_eq!(p.channels.iter().map(|c| c.l).collect::<Vec<_>>(), vec![0, 1], "Z = {z}");
            assert!(!p.semicore);
        }
    }

    /// The explicit no-core branch (spec §4.2 item 4): H and He have no core.
    #[test]
    fn hydrogen_and_helium_have_no_core() {
        for z in [1u8, 2] {
            let a = ground_state(z, SpinMode::Restricted).unwrap();
            let p = partition(&a).unwrap();
            report(&p);
            assert!(p.core.is_empty() && p.n_core == 0 && p.z_v == z as f64, "Z = {z}: {p:?}");
            assert!(core_density(&a, &p).iter().all(|&x| x == 0.0));
            assert!(p.channels.iter().any(|c| c.l == 0 && c.occupied));
            assert!(!has_core(z).unwrap());
        }
    }

    /// Spec §4.2 item 3 (Task 2 review S1): the neutral LSDA atom has no −1/r tail, so the 3d of Al–Ar
    /// is unbound and dropped, and no channel is unoccupied — s and p only, p local. A recipe or solver
    /// change that binds the 3d turns this red and comes to review (the S5 cap is then exercised on it;
    /// today it binds only for Be, `beryllium_unoccupied_p_is_capped`).
    #[test]
    fn al_through_ar_3d_is_unbound_and_dropped() {
        for z in 13u8..=18 {
            let a = ground_state(z, SpinMode::Restricted).unwrap();
            let p = partition(&a).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
            eprintln!("Z = {z}: dropped {:?}, channels {:?}", p.dropped, p.channels.iter().map(|c| (c.l, c.occupied)).collect::<Vec<_>>());
            assert_eq!(p.dropped, vec![2], "Z = {z}");
            assert!(p.channels.iter().all(|c| c.occupied), "Z = {z}: an unoccupied channel appeared");
        }
    }

    /// Review S5 where the cap binds: Be's l_max = 1 channel is the unoccupied 2p (bound in the neutral
    /// atom, r_peak beyond the 2s's), so at every f its r_c is the 2s's r_c — not f·r_peak,2p.
    #[test]
    fn beryllium_unoccupied_p_is_capped() {
        let a = ground_state(4, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let (s, pc) = (&p.channels[0], &p.channels[1]);
        assert!(s.occupied && s.l == 0 && !pc.occupied && pc.l == 1, "{p:?}");
        for f in crate::radial::LEDGER.f_ladder {
            let chans = crate::radial::tm::tm_channels(&a, &p, f).unwrap();
            let free = a.grid.r[a.grid.nearest(f * pc.r_peak)];
            eprintln!("Be f = {f}: r_c,s {:.4}; 2p (ε {:.6}, r_peak {:.4}): f·r_peak {free:.4} → r_c,p {:.4}", chans[0].r_c, pc.eps, pc.r_peak, chans[1].r_c);
            assert!(free > chans[0].r_c, "the cap does not bind at f = {f}");
            assert_eq!(chans[1].r_c, chans[0].r_c, "f = {f}");
        }
    }

    /// Spec §4.2 item 4: Al–Ar have the n = 2 core (1s 2s 2p, N_c = 10), valence 3s 3p, l_max = 2.
    #[test]
    fn al_through_ar_cores_are_n2() {
        for z in 13u8..=18 {
            let p = part(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
            report(&p);
            assert_eq!(p.core.iter().map(|s| (s.n, s.l)).collect::<Vec<_>>(), vec![(1, 0), (2, 0), (2, 1)], "Z = {z}");
            assert_eq!((p.n_core, p.z_v), (10, z as f64 - 10.0), "Z = {z}");
            assert_eq!(p.l_max, 2);
            assert!(!p.semicore);
        }
    }

    /// Spec §4.2 item 4: Na and Mg promote the n = 2 shell (semicore) and carry 2s and 3s in the s
    /// channel → refused until the ONCV stage.
    #[test]
    fn sodium_and_magnesium_are_refused_semicore() {
        for z in [11u8, 12] {
            match part(z) {
                Err(r @ Refusal::TwoStatesSameL { l: 0, .. }) => eprintln!("{r}"),
                other => panic!("Z = {z}: expected a two-s-state refusal, got {other:?}"),
            }
        }
    }

    /// ρ_core + ρ_valence is the reference atom's total density (both spins), pointwise.
    #[test]
    fn core_plus_valence_is_the_total_density() {
        for z in [1u8, 6, 8, 14] {
            let a = ground_state(z, SpinMode::Restricted).unwrap();
            let p = partition(&a).unwrap();
            let (c, v) = (core_density(&a, &p), valence_density(&a, &p));
            let worst = (0..a.grid.len()).map(|i| {
                let t = a.rho[0][i] + a.rho[1][i];
                (c[i] + v[i] - t).abs() / t.abs().max(1e-300)
            }).filter(|x| x.is_finite()).fold(0.0, f64::max);
            eprintln!("Z = {z}: max |ρ_c + ρ_v − ρ| / ρ = {worst:.2e}");
            assert!(worst < 1e-13, "Z = {z}: {worst:.2e}");
        }
    }

    fn report(p: &Partition) {
        let sh = |v: &[Shell]| v.iter().map(|s| format!("{}{}^{} ({:.6})", s.n, l_name(s.l), s.occ, s.eps)).collect::<Vec<_>>().join(" ");
        let ch = p.channels.iter().map(|c| format!("{}{}{} ε {:.6} r_peak {:.4}", c.n, l_name(c.l), if c.occupied { "" } else { "*" }, c.eps, c.r_peak))
            .collect::<Vec<_>>().join("; ");
        eprintln!("Z = {}: core [{}] valence [{}] N_c {} Z_v {} semicore {} l_max {} channels [{}] dropped {:?}",
                  p.z, sh(&p.core), sh(&p.valence), p.n_core, p.z_v, p.semicore, p.l_max, ch, p.dropped);
    }

    /// `Partition::spectrum` (Task 3, G2's all-electron count): per l its first entry is the channel's own
    /// reference state (the AE eigenvalue re-solved in the stored output potential: SCF lag ≲ 1e-9), a
    /// dropped l has none, and every entry is bound. Reports the higher bound states the neutral LSDA atom
    /// has (one weak (n+1)s).
    #[test]
    fn spectrum_above_the_core_starts_at_the_channel_state() {
        for z in (1u8..=2).chain(4..=10).chain(13..=18) {
            let p = part(z).unwrap();
            assert_eq!(p.spectrum.len(), p.l_max as usize + 1);
            for c in &p.channels {
                let e0 = p.spectrum[c.l as usize][0];
                assert!((e0 - c.eps).abs() < 1e-9, "Z = {z}, l = {}: spectrum starts at {e0} vs channel ε {}", c.l, c.eps);
            }
            for &l in &p.dropped { assert!(p.spectrum[l as usize].is_empty(), "Z = {z}: dropped l = {l} has bound states"); }
            assert!(p.spectrum.iter().flatten().all(|&e| e < 0.0));
            eprintln!("Z = {z:2}: spectrum above the core per l: {:?}", p.spectrum.iter().map(|v| v.iter().map(|e| format!("{e:.6}")).collect::<Vec<_>>()).collect::<Vec<_>>());
        }
    }
}
