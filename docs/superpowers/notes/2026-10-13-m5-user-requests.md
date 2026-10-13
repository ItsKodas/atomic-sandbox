# User requests captured during M3 play-testing (2026-10-13) — input to the M5 (tools) plan

Landed immediately on `m4-dynamics` (small, app-only): middle-drag orbit / right-drag pan
(00ebded); distance guides while dragging or placing (7e263e3); display aids — cell box, floor
grid with axes, placement-plane grid, 1 Bohr / 1 Å spacing (this commit).

## For M5

1. **Bond graph from the density, not from distance tables.** Two nuclei are bonded when the
   electron density along their internuclear segment never drops below a threshold (a bond
   critical point in Bader/QTAIM terms). Element-agnostic (rule 1), computed on the sim side after
   each solve (small readback or a GPU line-sampling kernel), published in the snapshot. Draw thin
   bond lines; connected components are molecules.
2. **Molecule labels.** Hill formula with charge (H₂, H₂O, LiH, H₃⁺) next to each component.
   Programmatic structural names where the bond graph defines them (binary hydrides/oxides/
   halides, *peroxide* = O–O single bond with both O further bonded, simple hydrocarbons), plus a
   small trivial-name table (water, ammonia, methane, hydrogen peroxide, …). The name table is
   display data like element colours — allowed; it never feeds physics.
3. **Isotopes** came up ("tritium"): currently every element uses its most abundant isotope's
   mass (`units::isotope_mass_amu`). A per-nucleus mass number is a legitimate M5 addition (mass
   is one of the two allowed per-element inputs); it only matters once M4 dynamics exist (isotope
   effect on vibration frequencies — a nice validation: H₂ vs D₂ ω ratio √2).
4. Honest expectations to put in the UI copy: H₃ is not a stable molecule (H₃⁺ is); bonds and
   names are *read off* the density, so they appear only after a solve converges.
