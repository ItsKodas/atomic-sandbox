# Orbital viewer: handoff (Tasks 6 and 7, 2026-10-16)

Branch `orbital-viewer`. Spec `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md`, plan
`docs/superpowers/plans/2026-10-15-orbital-viewer.md`, numbers `crates/validate/references.md`
"Orbital viewer". Tasks 0–5 built the numerics, the GPU products, the data path, the display
eigensolve and the oracles; Task 6 the rendering and Task 7 the controls, the honesty text and
these docs. Task 8 (core shells reconstructed for display) waits for M7.

## What exists

- **Rendering** (`render`): `VolumeUniform` modes 2 (signed isosurface) and 3 (signed volume) with
  `colour_pos`/`colour_neg`; `ColourVolumePass` (a second pipeline in `volume.rs`, shader
  `volume_colour.wgsl`, sharing the ray setup of `volume.wgsl`) for the rgba16float colour volume
  with a previous/current pair, additive or weighted; `MODE_DOTS` moved to 4; the dots sample a
  signed field by |f| and colour by sign. The orbital colour palette stays in
  `qm_gpu::orbitals::ORBITAL_PALETTE` (`providers` re-exports it): `qm-gpu` cannot depend on
  `render` (the producer in `providers` would otherwise need a rendering crate under the force
  boundary, rule 5), so the Task 2 ruling "move it to render" is superseded.
- **App** (`app`): `gfx::Painter` holds two layers (the density and the orbital layer) and binds
  the passes lazily per the `show` mode; an orbital payload crossfades with its predecessor of the
  same request and grid and snaps otherwise; `viewer::OrbitalShown` mirrors `DensityShown`;
  `ui::ShowMode`/`UiState::orbital_request`; `ui/dock/orbitals.rs` (the tab: list, controls, the
  honesty constants, pure view-model functions the tests read); the legend chip; menus, search,
  actions with script words; `--ui` keys. The proportional font falls back to Hack for glyphs it
  lacks (the arrows).
- **Carried items, as resolved**
  - `UiState::run_settings()` carries `DynCommand::SetOrbitalView`; `viewer::start` holds the
    unoccupied rows back until the new worker's first payload.
  - **Unoccupied eigensolve in play: a latency cap, not a ban.** `UNOCCUPIED_PLAY_MAX_WALL` = 0.25 s
    joins the 10 % duty budget (`providers::orbitals::UnoccupiedCadence`), so in play it runs only
    if its last wall was ≤ 0.25 s: the warm 49³ solve (0.14 s) qualifies, every h = 0.15 grid
    (1.7–10 s) is paused-only. Chosen over "forbid in play" because small grids keep live
    unoccupied rows and the stall is bounded and interruptible; the tab shows "computing unoccupied
    orbitals…" whenever the request is not yet answered, and "updated when paused at this grid size".
  - Estimated-mode refresh retry: `worker::orbital_retry_due` asks the background solver for the
    refresh again, once per payload, when an interrupted side payload and an idle worker coincide.
  - Box-state wording by the sign of ε; rotated rows labelled "eigensolve residual of the set"
    beside the spread; "orbitals from t = … fs (last DFT solve). The estimated solver has no
    electrons"; the CPU teacher's text and spec §7 row; the `dead_code` attribute on
    `Command::OrbitalView` is gone.
  - HOMO, LUMO and the gap come from `OrbitalInfo::eps_range` (new in `qm::orbitals::Row` and
    `sim::OrbitalInfo`: the lowest and highest eigenvalue of the row's set), never from a rotated
    member's Rayleigh quotient. Rows within 1e-4 Ha of the HOMO (a closed shell's ↑ and ↓) are all
    tagged.
  - `sim::TRACK_HOMO`: asking for "the HOMO" by name is not a vanished selection, so the provider
    says nothing; the viewer adopts the track it resolved to and re-sends once.
  - The seven Task 5 minors: `h2_signs_…` sets the density extrapolation as `GpuKs` does (M1); the
    H and Li gaps decomposed in `references.md` (M2, M3); the 1.6 Bohr wording (M4); Li
    ∫|m| − ∫m > 1e-4 asserted (M5); `ORBITALS_SEED` accepts `0x…` and panics on garbage (M6);
    `INTERACTIVE_TOL` names its source (M7).

## Costs (RX 7900 XTX, release; reported, never asserted)

| | 49³ | 57³ | 81³ | 97³ |
|---|---|---|---|---|
| display update, K = 10, product none (lone atom / molecule), ms | 1.3 / 1.0 | 1.8 / 1.0 | — | 2.3 / 1.3 |
| … with the signed-orbital texture, ms | 1.0 / 0.8 | 1.6 / 0.9 | — | 2.2 / 1.1 |
| readbacks per update | 1 | 1 | 1 | 1 |
| view VRAM, m = 0 / m = 4 / m = 4 colour, GB | 0.011 / 0.094 / 0.100 | 0.016 / 0.140 / 0.149 | 0.041 / 0.359 / 0.384 | 0.067 / 0.589 / 0.633 |
| a solve itself, GB (formula) | 0.145 | 0.215 | 0.552 | 0.906 |

(`qm-gpu` `diag_orbital_update_phases --ignored`; `providers` `the_orbital_view_is_charged_to_the_vram_budget`.)

**The unoccupied solve's wall against its budget** (Task 5, h = 0.15, tol 1e-5, m = 4): H₂ 81³
1.7–1.8 s = 8.6–9.1 warm SCFs; H₂O 95³ 9.1–10.3 s = 7–8 warm SCFs. The 10 % budget alone needs
≥ 10 × its wall of SCF time between runs (H₂ ≈ 17 s, H₂O ≈ 91 s); the 0.25 s cap now keeps both
paused-only. Warm 49³: 0.14 s, inside the cap.

**Frame times, 97³ in a 1400 × 900 headless frame** (mean of 20, `render::volume::orbital_tests::frame_times_at_97_cubed`):
volume 0.8–0.9 ms, isosurface 0.5–0.6, signed isosurface 0.5, signed volume 0.6–0.7, colour weighted 0.5,
colour additive 0.65–0.7, 200 k dots 0.8–0.85 (the brief's 2 ms), signed dots 0.7–0.8.

**Play with the view on** (H₂, Accurate, GPU, 6 s): the same trajectory (t 13.3 fs, step 53,
E conserved spread 3.10e-5 Ha) and 2.2–2.6 fs/s with and without it.

## Review round 1 (Tasks 6–7)

- **I1** The inversion warning now sits directly under the levels line, above the rows; the levels
  line, the HOMO tag (amber, "shell order inverted: not the physical HOMO") and its tooltip say the
  inverted atom's "HOMO" is not physical; the HOMO tooltip itself no longer calls the eigenvalue
  "physically meaningful" without the LSDA / smeared-nucleus caveat. Molecules whose heaviest
  nucleus has Zσ ≥ 0.5 get a smeared-core note in the same place (the review's minor 5).
- **I2** Cause: the estimated solver's background teacher restarts its tracks after an isolated-atom
  job (the cloud's superposition asks for one per species), so the HOMO the viewer had adopted no
  longer resolves and the provider's fallback note ("selected orbital no longer exists") is raised
  for a selection the user never made; the shot caught it because the screenshot waits only for
  the payload that matches the adopted pick, not for the next one. Fix: `UiState::orbital_pick_adopted`
  (set when the viewer adopts the HOMO the producer resolved for the placeholder, cleared by a
  picked row); the viewer re-adopts after a restart and the tab drops that note for an adopted
  pick; a user's vanished pick keeps its (true) note. Test:
  `the_viewer_adopts_the_resolved_orbital_and_releases_the_unoccupied_rows`.
- Minors: signed-iso pixel test tightened to ≤ 8 pixels and < 64; legend iso clause only in
  isosurface mode; "not split by the grid (< 0.005 mHa)" below the display resolution; widget
  hover-text test (`the_widgets_show_the_honesty_text_on_hover`); `let _ = ctx` removed;
  the palette ruling worded as superseded; the `dots_by_orbital` default (true) in the README.
- **Still owed for the merge gate:** the worker-level test of `Worker::orbital_retry` against the
  real auditor (only the predicate `orbital_retry_due` is tested); the dots' χ² tests are the
  preview port's synthetic trilinear fields (constant, ramp, sharp peak) plus the signed-field one
  — not literally H 1s and Ne 2p_x, the property is the same; the rule-6 suites above.

## Open items

1. **The all-orbitals colour view of a many-electron atom is a haze.** Ne's five orbital pairs
   overlap at the nucleus, additive emission of five hues is pastel, and the smeared 1s is two
   grid cells wide. It is what spec §5.6 defines (rgb = Σ f c s over the included orbitals); to
   make shells legible the next step is a per-orbital isosurface shell or sharpening the weights
   (s^p), a spec change. The single-orbital views and water read well.
2. **The 2s node is hidden in isosurface mode** (the first sample at |s| ≥ iso is the outer
   surface); the volume mode shows it. Known from the preview.
3. **Estimated mode: every changed view asks for one warm side SCF** (Task 3 review M6). A
   "re-display without a solve" job would remove it when the newest side solve is at the current
   geometry.
4. **The orbital view was never driven with a mouse in an OS window** (checklist note,
   `2026-10-13-m2-viewer-checklist.md`, last section): slider drags, the include boxes, the strip
   icon and the flicker of a vibrating molecule's signs are judged by widget tests, headless
   frames and the oracles only.
5. **Dots of a signed field**: a nodal cell is slightly over-filled (|interpolated| is not the
   interpolant of |·|); an orbital is therefore drawn as ψ̃² in dots mode (smooth through its
   nodes), and only the spin density uses |m|.
6. **Not run in this task** (rule 6 gates for the milestone merge): `cargo test -p validate
   --release` in full, the dynamics and surrogate suites (Task 5 ran them at 8239e9f; this task
   changed `sim::OrbitalInfo`, `providers::orbitals` (the cadence cap, `TRACK_HOMO`) and the
   worker only).
7. **Task 8** (core shells): `CORE_TAG` ("core: reconstructed for display") is defined in
   `ui/dock/orbitals.rs` and pinned by a test; the rows, the analytic shader path and the physical
   scaling that includes the core belong to it.
