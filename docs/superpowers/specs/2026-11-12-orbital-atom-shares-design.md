# Orbitals by atom — design

Status: for the user's review (conversation 2026-11-11/12; builder check done).
Builds on the orbital viewer (`docs/superpowers/specs/2026-10-15-orbital-viewer-design.md`) and
the partial charges (`crates/sim/src/charges.rs`, Hirshfeld).

## Why

The user wants to see which orbitals are mostly on a given atom: "I want to see which orbitals
are mostly on that atom". Today the Orbitals list is one flat list per spin (or one ↑↓ list for a
closed shell, user request 2026-11-11), so in a molecule there is no way to tell which rows belong
to the oxygen and which are bonds.

## The physics, stated plainly

A molecular orbital belongs to no single atom; it is spread over the molecule. "Mostly on that
atom" therefore needs a split of space between the atoms. We use the one the app already uses for
partial charges: the **Hirshfeld (stockholder) weights** w_A(r) = ρ_A⁰(|r − R_A|) / Σ_B ρ_B⁰(|r − R_B|),
built from this solver's own isolated-atom references (`crate::references`, CLAUDE.md rule 1: no
per-element data). An orbital's share on atom A is

```text
s_iA = ∫ w_A(r) |ψ_i(r)|² d³r  ≈ h³ Σ_nodes w_A(r_n) |ψ_i(r_n)|²,   Σ_A s_iA = ∫ |ψ_i|² = 1.
```

The share is a property of the chosen split, not a measurable quantity (rule 7, physics honesty):
the UI says so in the row's tooltip, as the partial charges do. Display only: nothing here feeds a
solve, a force, the surrogate or a sample.

## Goals

1. **Every orbital row shows where it lives:** the largest share and its atom (e.g. `O #0 92 %`);
   hover shows every atom with a share ≥ 1 %, largest first, and the Hirshfeld note.
2. **The list is grouped by atom:** an orbital whose largest share is a clear majority
   (s ≥ 0.55, see the calls below) is listed under that atom's heading (`O #0`, `H #1`, …, the same atom names the Selection
   panel uses). Orbitals with no majority go under a **shared** heading naming the atoms that carry
   them (e.g. `shared · H #0–H #1`).
   Within a group, rows keep today's order (by energy) and today's ↑ / ↓ / ↑↓ channel structure
   and occupied / unoccupied split.
3. **Selection lights up the selected atoms' orbitals:** with one or more atoms selected in the 3D
   view, a row whose combined share on the selected atoms is at least 0.55 stays at full
   strength; every other row is **dimmed but stays in the list** and stays clickable (user
   decision 2026-11-12). With nothing selected, nothing is dimmed. A bond's shared orbital lights
   up when both its atoms are selected.
4. **A one-nucleus scene is unchanged** (every share is 100 % on the one atom; no grouping
   headings, no share column).

## Calls made after the builder check (Claude, for the user to check)

- **Degenerate sets share one answer.** The display rotates each degenerate set to a steady
  orientation, and a single member's share depends on that rotation. So every member of a set
  (`OrbitalInfo::cluster`) shows the set's average share, (1/d) Σ_{i∈set} s_iA. That average does
  not depend on the rotation, and the set stays together under one heading (CH₄'s t₂, NH₃'s e).
- **Near-ties read as shared, with no memory.** Grouping depends only on the current shares. An
  orbital goes under an atom only when that atom's share is at least 0.55, so exact and near
  50/50 orbitals (H₂'s σ_g) read `shared` and do not flicker between two atoms on grid noise. The
  atoms named in a `shared · …` heading are those with a share of at least 0.25, largest first;
  equal shares are ordered by atom number. The selection rule (goal 3) uses the same 0.55. This
  is a display margin, not physics, and the tooltip states it.
- **Every backend; the shares check themselves.** The DFT backends (GPU and CPU) and the estimated
  solver are all in scope. Shares are computed for the orbitals' own geometry and solve, against
  the free-atom references of that solve's surface (its `ReferenceKey`). They are shown only
  when those references exist and the shares' own sum Σ_A s_iA = 1 holds to the same `SUM_TOL`
  the charges use. They never borrow another step's charges. When the references are missing,
  the note says why.
- **Box states and stale rows get no share.** A row tagged as a state of the box (spread over the
  Dirichlet box) or as solved at an earlier geometry is listed under its own heading (`box
  states` / `from an earlier geometry`) with no share column. Splitting a box state's far tails
  between atoms would present an artefact as a fact.

## Non-goals

- Changing any solve, force, charge or the orbitals' shapes; a Mulliken/Löwdin or Bader split;
  per-orbital charges; picking atoms from the list (the list does not change the selection).
- Recomputing shares while the orbitals are stale: shares belong to the orbitals they were
  computed with.

## Anti-goals (a failure even if it works)

- Shares presented as measured facts, or hidden caveats.
- A list that jumps or re-sorts as the user selects atoms (dimming only; grouping depends on the
  shares, never on the selection).
- Any per-element threshold or table.

## Constraints

- CLAUDE.md rules 1 (no per-element data), 3 (atomic units inside; % only at the UI), 4 (grid in
  f32 on the GPU; the share sums are reductions — use compensated/pairwise summation), 5
  (`qm` knows nothing of the UI; `sim` does not know what computes forces), 7 (honesty).
- The shares reuse the partial charges' Hirshfeld weights and references; where the charges are not
  available (references still being solved, the charges' `sum_error` check failing), the shares are
  not shown and the list is as today with a one-line note ("atom shares need the atom references:
  being solved" / the charges' own reason).
- The orbitals come from the SCF (occupied) and the display eigensolve (unoccupied); both live as
  fields on the GPU (f32). Shares are computed once per new set of orbitals, off the stepping path,
  and must not delay a step or an edit (an interrupted display solve leaves shares absent).
- Paired ↑↓ rows use the ↑ member's shares.
- Every new UI control keeps a Ctrl K search anchor; display state only, nothing becomes an
  `Action` (selection is already tracked).

## What's left to the builder

Where the reduction runs (GPU kernel against the existing weights vs a CPU pass over a readback),
how the shares travel to the UI (a field on `OrbitalInfo` or a parallel table in the orbital
payload), the exact layout of headings and the share column, the dimming strength (theme's faint
ink), tests and their tolerances (Σ_A s = 1 to rounding; symmetric H₂ 50/50; separated atoms
≈ 100 % on their own atom; an O lone pair mostly on O), and where in `validate` the oracle sits
(orbital display suite).
