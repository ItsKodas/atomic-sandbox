# UI redesign: Appendix A checklist (2026-10-14)

The acceptance checklist of the UI redesign: every row of Appendix A of the mockup
(`docs/superpowers/notes/2026-10-14-ui-mockup.html`, 148 controls and readouts of the old side
panel, the atom card, the overlays and the mouse/keyboard input) with where it lives now. File
references are relative to `crates/app/src/` and point at the line that draws (or records) the
item; this table is generated from a list of literal anchors, so a row only reads `done` when its
anchor text exists at that line. Tooltip texts of the old panel are kept verbatim on the moved
controls.

**148 / 148 rows done.**

Abbreviations as in the mockup: TB = bottom toolbar, Dock ▸ X = right dock tab, Menu = top menu,
Overlay = drawn in the 3D view.

| # | Old item | New place | Status | Where (file:line) |
|---|---|---|---|---|
| 1 | heading "Atomic Sandbox" | Top bar brand + scene chip | done | `ui/menus.rs:93`, `ui/menus.rs:104` |
| 2 | status line + spinner | Overlay status chip (all states kept) | done | `ui/overlays.rs:66`, `ui/overlays.rs:92` |
| 3 | UI error (red) | Status chip red + Log (error) | done | `ui/overlays.rs:93`, `ui/logbook.rs:160` |
| 4 | "paused by a solver failure" | Status chip red + Log + strip count | done | `ui/overlays.rs:97`, `ui/logbook.rs:171` |
| 5 | "refused: …" (amber) | Log tab "last refusal" line, in place while the driver holds it + 6 s toast + Log entry | done | `ui/dock/log.rs:10`, `ui/logbook.rs:177`, `ui/overlays.rs:168` |
| 6 | neutral status (saved / loaded / switched) | Toast + Log (status) | done | `ui/logbook.rs:157`, `ui/overlays.rs:168` |
| 7 | solver radio · CPU | TB Solver segmented | done | `ui/toolbar.rs:368` |
| 8 | solver radio · GPU | TB Solver segmented, "no GPU adapter" | done | `ui/toolbar.rs:374` |
| 9 | solver radio · estimated | TB Solver segmented, tooltip names the teacher | done | `ui/toolbar.rs:28` |
| 10 | honesty line | Dock ▸ Model first line + i; Help ▸ Physics, honestly | done | `ui/dock/model.rs:18`, `ui/menus.rs:526` |
| 11 | this step (surrogate / fallback / DFT) | TB Solver sub-line + status chip pill + Dock ▸ Model | done | `ui/dock/model.rs:269`, `ui/toolbar.rs:392`, `ui/overlays.rs:121` |
| 12 | uncertainty vs bound | Dock ▸ Model (bar + numbers, tooltip kept) | done | `ui/dock/model.rs:272`, `ui/dock/model.rs:365` |
| 13 | trusted (five variants) | TB trust light + Dock ▸ Model + strip dot | done | `ui/mod.rs:466`, `ui/dock/model.rs:289`, `ui/dock/mod.rs:59` |
| 14 | samples (+ refit pending) | Dock ▸ Model | done | `ui/dock/model.rs:296` |
| 15 | model epoch | Dock ▸ Model + TB sub-line | done | `ui/dock/model.rs:299`, `ui/toolbar.rs:430` |
| 16 | last fit | Dock ▸ Model | done | `ui/dock/model.rs:302` |
| 17 | last refit | Dock ▸ Model | done | `ui/dock/model.rs:309` |
| 18 | fallbacks % of last 100 | Dock ▸ Model (bar) + TB sub-line | done | `ui/dock/model.rs:316`, `ui/toolbar.rs:430` |
| 19 | teacher gap | Dock ▸ Model | done | `ui/dock/model.rs:327` |
| 20 | last audit (verdict, gap, ms, lag, in flight) | Dock ▸ Model; audit lag also in Readouts | done | `ui/dock/model.rs:335`, `ui/dock/readouts.rs:168` |
| 21 | mismatch work | Dock ▸ Model | done | `ui/dock/model.rs:356` |
| 22 | provenance mismatch warning | Dock ▸ Model warning slot + Log | done | `ui/dock/model.rs:36`, `ui/logbook.rs:231` |
| 23 | autoload refused warning | Dock ▸ Model warning slot + Log | done | `ui/dock/model.rs:39`, `ui/logbook.rs:238` |
| 24 | "Forget it and sample this scene" | Dock ▸ Model, in the warning | done | `ui/dock/model.rs:40` |
| 25 | sampling progress bar | Sampling card + status chip + Dock ▸ Model | done | `ui/dock/model.rs:128`, `ui/overlays.rs:142` |
| 26 | last teacher energy · skipped · failed | Sampling card + Dock ▸ Model | done | `ui/dock/model.rs:134` |
| 27 | last pass summary | Dock ▸ Model + Log (pass) | done | `ui/dock/model.rs:53`, `ui/logbook.rs:213` |
| 28 | long first-pass offer (amber) | Offer card in the view + Dock ▸ Model | done | `ui/dock/model.rs:140`, `ui/overlays.rs:27` |
| 29 | offer "Start anyway (≈ t)" | Offer card (and Model tab) | done | `ui/dock/model.rs:153` |
| 30 | DFT audit every n | Dock ▸ Model ▸ Audits | done | `ui/dock/model.rs:69` |
| 31 | "(solver: n)" | Dock ▸ Model, beside the field | done | `ui/dock/model.rs:81` |
| 32 | audits-off unsafe warning | Dock ▸ Model, under the field | done | `ui/dock/model.rs:85` |
| 33 | plan · perturbations | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:232` |
| 34 | plan · σ_r | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:236` |
| 35 | plan · hot burst steps | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:242` |
| 36 | plan · hot burst T | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:243` |
| 37 | plan · scan points / pair | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:248` |
| 38 | plan · seed | Dock ▸ Model ▸ Sampling plan | done | `ui/dock/model.rs:254` |
| 39 | Start sampling / Sample more / armed "Start anyway" | Dock ▸ Model primary button (two clicks above 2 min) | done | `ui/dock/model.rs:158`, `ui/actions.rs:231` |
| 40 | plan cost "n solves, ≈ t" | Dock ▸ Model, beside the button | done | `ui/actions.rs:246` |
| 41 | Cancel | Sampling card + Dock ▸ Model | done | `ui/dock/model.rs:180`, `ui/overlays.rs:158` |
| 42 | Forget model (two clicks within 4 s) | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:215`, `ui/menus.rs:223` |
| 43 | Save model | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:196` |
| 44 | Load model | File ▸ Estimated model + Dock ▸ Model | done | `ui/dock/model.rs:203` |
| 45 | model path label | File ▸ Estimated model header + Dock ▸ Model | done | `ui/menus.rs:220`, `ui/dock/model.rs:101` |
| 46 | Play / Pause | TB ▸ Time primary button (Space; sampling reason kept) | done | `ui/toolbar.rs:219` |
| 47 | Step | TB ▸ Time | done | `ui/toolbar.rs:238` |
| 48 | Rewind | TB ▸ Time | done | `ui/toolbar.rs:220` |
| 49 | speed slider (steps/tick, log) | TB ▸ Speed (slider + typed field) | done | `ui/toolbar.rs:102` |
| 50 | "watchable (≈ fs/s)" | TB ▸ Speed sub-line | done | `ui/toolbar.rs:115` |
| 51 | measured fs/s (last 2 s) | TB ▸ Speed (large) + Readouts tile and Time and cost | done | `ui/toolbar.rs:108`, `ui/dock/readouts.rs:76` |
| 52 | fs/s this tick | TB ▸ Speed sub-line + Readouts | done | `ui/toolbar.rs:121`, `ui/dock/readouts.rs:143` |
| 53 | steps last tick | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:145` |
| 54 | ms / step | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:147` |
| 55 | dt | TB ▸ dt + Physics menu | done | `ui/menus.rs:359`, `ui/toolbar.rs:129` |
| 56 | t = fs, step N | Status chip + Readouts | done | `ui/overlays.rs:126`, `ui/dock/readouts.rs:138` |
| 57 | dt warning (amber) | TB dt field amber + Readouts ▸ Health + Log | done | `ui/menus.rs:361`, `ui/dock/readouts.rs:187` |
| 58 | T | Readouts tile + status chip | done | `ui/dock/readouts.rs:47`, `ui/overlays.rs:129` |
| 59 | KE (nuclei) | Readouts ▸ Energies (Ha / eV / both) | done | `ui/dock/readouts.rs:94` |
| 60 | E_KS | Readouts ▸ Energies | done | `ui/dock/readouts.rs:95` |
| 61 | E total | Readouts ▸ Energies (bold) | done | `ui/dock/readouts.rs:96` |
| 62 | E conserved | Readouts ▸ Energies | done | `ui/dock/readouts.rs:98` |
| 63 | work done by tools | Readouts ▸ Energies | done | `ui/dock/readouts.rs:105` |
| 64 | model refit (estimated) | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:116` |
| 65 | model force mismatch (estimated) | Readouts ▸ Energies, indented | done | `ui/dock/readouts.rs:122` |
| 66 | drift (coloured) | Readouts tile + strip dot (1e-4 / 1e-3) | done | `ui/dock/readouts.rs:64`, `ui/dock/mod.rs:50` |
| 67 | per step (model ms / SCF steps, readbacks, ms) | Readouts ▸ Time and cost | done | `ui/dock/readouts.rs:153` |
| 68 | sparkline (E total by source, T, fallback dots) | Dock ▸ Graph (large) + mini in Readouts | done | `ui/dock/graph.rs:44`, `ui/dock/readouts.rs:180` |
| 69 | sparkline ranges + "last N fs" | Dock ▸ Graph | done | `ui/dock/graph.rs:125` |
| 70 | source legend | Dock ▸ Graph | done | `ui/dock/graph.rs:131` |
| 71 | box warning (amber / red) | Readouts ▸ Health + Log + strip dot | done | `ui/dock/readouts.rs:183`, `ui/logbook.rs:182` |
| 72 | dynamics notes (last 3) | Readouts ▸ Health (the 3 newest, as before) + Log (all) | done | `ui/dock/readouts.rs:194`, `ui/logbook.rs:92` |
| 73 | thermostat on/off | Physics menu + status chip "CSVR T K" | done | `ui/menus.rs:387`, `ui/overlays.rs:131` |
| 74 | thermostat target T | Physics menu | done | `ui/menus.rs:392` |
| 75 | thermostat τ | Physics menu | done | `ui/menus.rs:396` |
| 76 | gravity "g = 9.8 m/s² (−y)" | Physics menu | done | `ui/menus.rs:415` |
| 77 | gravity "off" | Physics menu | done | `ui/menus.rs:411` |
| 78 | COM correction | Physics menu | done | `ui/menus.rs:442` |
| 79 | field magnitude readout | Physics menu, under the field | done | `ui/menus.rs:431` |
| 80 | preset combo | File ▸ Presets + scene chip | done | `ui/menus.rs:176`, `ui/menus.rs:138` |
| 81 | Reset | File ▸ Presets + scene chip | done | `ui/menus.rs:185` |
| 82 | tool Orbit | TB ▸ Tool | done | `ui/toolbar.rs:256` |
| 83 | tool Place | TB ▸ Tool | done | `ui/toolbar.rs:257` |
| 84 | tool Select | TB ▸ Tool | done | `ui/toolbar.rs:258` |
| 85 | element to place (H–Ne) | TB ▸ element chips (pick → Place); compact flyout | done | `ui/toolbar.rs:274`, `ui/toolbar.rs:300`, `ui/actions.rs:87` |
| 86 | "no selection" hint | TB ▸ Selection placeholder + Dock ▸ Selection | done | `ui/toolbar.rs:161`, `ui/dock/selection.rs:12` |
| 87 | "selected: X #i at (x, y, z) Bohr" | TB ▸ Selection label + Dock ▸ Selection | done | `ui/toolbar.rs:152`, `ui/dock/selection.rs:24` |
| 88 | Delete selected | TB ▸ Selection + Dock ▸ Selection | done | `ui/toolbar.rs:355`, `ui/dock/selection.rs:47` |
| 89 | Focus selected | TB ▸ Selection + Dock ▸ Selection | done | `ui/toolbar.rs:346`, `ui/dock/selection.rs:44` |
| 90 | pinned | TB ▸ Selection Pin + Dock ▸ Selection | done | `ui/toolbar.rs:338`, `ui/dock/selection.rs:38` |
| 91 | hint "… edits pause the run and restart its time" | Scene menu, top line | done | `ui/menus.rs:234` |
| 92 | charge | Scene ▸ Electrons + scene chip | done | `ui/menus.rs:238` |
| 93 | multiplicity auto | Scene ▸ Electrons | done | `ui/menus.rs:249` |
| 94 | multiplicity value | Scene ▸ Electrons | done | `ui/menus.rs:250` |
| 95 | σ auto = h | Scene ▸ Electrons | done | `ui/menus.rs:260` |
| 96 | σ value | Scene ▸ Electrons | done | `ui/menus.rs:262` |
| 97 | grid n | Scene ▸ Grid (odd n only) | done | `ui/menus.rs:280` |
| 98 | grid h | Scene ▸ Grid | done | `ui/menus.rs:287` |
| 99 | cost readout | Scene ▸ Grid | done | `ui/menus.rs:296` |
| 100 | Apply grid | Scene ▸ Grid (enabled with a pending edit) | done | `ui/menus.rs:289` |
| 101 | density volume / isosurface | View ▸ Density + quick switch in the view | done | `ui/menus.rs:321`, `ui/overlays.rs:199` |
| 102 | gamma | View ▸ Density (disabled for isosurface) | done | `ui/menus.rs:327` |
| 103 | opacity | View ▸ Density (disabled for isosurface) | done | `ui/menus.rs:328` |
| 104 | iso (× ρ max) | View ▸ Density (disabled for volume) | done | `ui/menus.rs:329` |
| 105 | cell | View ▸ Display aids | done | `ui/menus.rs:340` |
| 106 | floor grid | View ▸ Display aids | done | `ui/menus.rs:341` |
| 107 | axes | View ▸ Display aids | done | `ui/menus.rs:342` |
| 108 | smooth motion | View ▸ Display aids | done | `ui/menus.rs:350` |
| 109 | placement plane | View ▸ Display aids | done | `ui/menus.rs:345` |
| 110 | spacing 1 Bohr / 1 Å | View ▸ Display aids | done | `ui/menus.rs:348` |
| 111 | E_KS (breakdown) | Readouts ▸ Energy breakdown and solver (collapsible) | done | `ui/dock/readouts.rs:228` |
| 112 | kinetic (electrons) | same collapsible | done | `ui/dock/readouts.rs:229` |
| 113 | external | same collapsible | done | `ui/dock/readouts.rs:230` |
| 114 | Hartree | same collapsible | done | `ui/dock/readouts.rs:231` |
| 115 | xc | same collapsible | done | `ui/dock/readouts.rs:232` |
| 116 | nuclear | same collapsible | done | `ui/dock/readouts.rs:233` |
| 117 | note "smeared nuclei (σ = …)" | same collapsible | done | `ui/dock/readouts.rs:236` |
| 118 | surrogate / fallback breakdown note | same collapsible | done | `ui/dock/readouts.rs:238` |
| 119 | GPU f32 scatter note | same collapsible | done | `ui/dock/readouts.rs:243` |
| 120 | SCF steps | same collapsible | done | `ui/dock/readouts.rs:251` |
| 121 | eigen residual (final) | same collapsible | done | `ui/dock/readouts.rs:254` |
| 122 | Poisson residual | same collapsible | done | `ui/dock/readouts.rs:255` |
| 123 | solve (ms, readbacks, engine) | same collapsible | done | `ui/dock/readouts.rs:259` |
| 124 | grid | same collapsible | done | `ui/dock/readouts.rs:263` |
| 125 | ρ max | same collapsible | done | `ui/dock/readouts.rs:267` |
| 126 | evaluation notes | same collapsible + Log | done | `ui/dock/readouts.rs:270`, `ui/logbook.rs:208` |
| 127 | evaluation warnings (amber) | Readouts ▸ Health + Log | done | `ui/dock/readouts.rs:191`, `ui/logbook.rs:205` |
| 128 | scene path field | File ▸ Scene file | done | `ui/menus.rs:197` |
| 129 | Save | File ▸ Scene file | done | `ui/menus.rs:199` |
| 130 | Load (+ .model sidecar) | File ▸ Scene file (sidecar rule and "sampling was running" kept) | done | `ui/menus.rs:206`, `ui/actions.rs:156` |
| 131 | atom card title | Overlay card top-right of the view + Dock ▸ Selection; card toggle (new) | done | `atom_card.rs:59`, `ui/dock/selection.rs:85` |
| 132 | "nucleus #i: p protons · n neutrons" | Card + Dock ▸ Selection | done | `atom_card.rs:105`, `ui/dock/selection.rs:27` |
| 133 | "z electrons (neutral atom), shells …" | Card + Dock ▸ Selection | done | `atom_card.rs:73`, `ui/dock/selection.rs:30` |
| 134 | Bohr schematic drawing | Card | done | `atom_card.rs:108` |
| 135 | "schematic — not to scale" | Card | done | `atom_card.rs:141` |
| 136 | shared-electrons caveat | Card (line + i) + Selection tab + Help | done | `atom_card.rs:82`, `ui/dock/selection.rs:86` |
| 137 | cell box (amber wire) | Overlay, unchanged (toggle #105) | done | `overlay.rs:39` |
| 138 | floor grid | Overlay, unchanged (toggle #106) | done | `overlay.rs:42` |
| 139 | axes | Overlay + corner gizmo (new) (toggle #107) | done | `overlay.rs:47`, `ui/overlays.rs:224` |
| 140 | placement plane (blue) | Overlay in Place mode (toggle #109) | done | `overlay.rs:50` |
| 141 | distance guides | Overlay (through the frame's viewport, clipped to it) + list in Dock ▸ Selection | done | `viewer.rs:658`, `ui/dock/selection.rs:58` |
| 142 | placement marker cross | Overlay | done | `viewer.rs:664` |
| 143 | drag spring ring + tether | Overlay | done | `viewer.rs:634` |
| 144 | pin square | Overlay | done | `viewer.rs:622` |
| 145 | faded-density label | Overlay bottom centre of the view (centred on the viewport the scene now renders into) | done | `ui/overlays.rs:41` |
| 146 | Space: play / pause | Unchanged; shown under TB ▸ Time | done | `main.rs:379`, `ui/toolbar.rs:94` |
| 147 | Delete: remove selected | Unchanged; shown in Selection tab and Help | done | `main.rs:369`, `ui/dock/selection.rs:54` |
| 148 | mouse orbit / pan / zoom | Unchanged, in the viewport's frame (pan scaled by its height); Help ▸ Controls | done | `main.rs:421`, `ui/menus.rs:506` |

## Deviations from the mockup's wording (coordinator rulings, fix round 1)

- **Row 5, refusals.** The mockup moves them to a toast plus a Log entry. The old panel showed the
  refusal in place for as long as the driver held it, so that place is kept: the Log tab's
  `last refusal` line (`ui/dock/log.rs` `last_refusal`) shows `DynSnapshot::last_rejection`
  until the next accepted command clears it. The 6 s toast and the Log entry stay as well, so nothing is lost.
- **Row 72, dynamics notes.** Readouts ▸ Health shows the three newest notes, as the old panel did.
  The mockup says "newest". Ruling: keep more information, not less. All notes are also in the Log.

## Fix round 1: the 3D view renders into the central viewport

The scene used to render full-window under the panels, which put it off-centre. It now renders into
the rect the top bar, toolbar and dock leave. One `render::camera::Viewport { origin, size }`
(physical pixels) is computed per frame (`viewer.rs` `viewport_of`). It drives:
- the render pass viewport and scissor (`Renderer::render_in`);
- the camera aspect and the shaders' pixel → NDC mapping (`CameraUniform::origin`);
- every mouse ↔ world path: `tools::pick`, `place_point`, `drag_point` and `project`, the
  distance guides, the drag spring and the pin squares (clipped to the view), and pan scaling.

Presses outside the viewport are ignored. The `--screenshot` frame is still the full window
(chrome + viewport).

## Not in Appendix A, new in this branch (approved conveniences)

- Ctrl K control search (`ui/search.rs`): fuzzy match over every control's name; picking a result
  opens its menu or dock tab (and collapsible), scrolls to it and outlines it.
- Ha / eV / both toggle for the Readouts energies (`ui/dock/mod.rs`, `ui/mod.rs` `energy_cells`).
- Log tab with filters and the strip's unread error / refusal counter (`ui/logbook.rs`,
  `ui/dock/log.rs`); toasts for status lines and refusals (`ui/overlays.rs` `toast`).
- Picking an element chip switches the tool to Place (`ui/actions.rs` `Action::Element`).
- Esc deselects (`ui/mod.rs` `keys`; not while a text field, a menu or the search has it).
- The atom card can be hidden (Selection tab).
- Volume / Isosurface quick switch and the axes gizmo (`ui/overlays.rs` `view_switch`, `gizmo`).
- Dock strip peeks (hover a tab button for its key numbers) and status dots.
- Help windows with the long-form "Physics, honestly" texts (`ui/menus.rs` `help_text`).

## Queued items (Appendix B): reserved, not implemented

Grep `SEAM(Q` for the seams. Each is a greyed, disabled control with a `QUEUED` tag.

| # | Item | Seam |
|---|---|---|
| Q1 | draft mode toggle | `ui/menus.rs` Physics ▸ Estimated solver; `ui/dock/model.rs` last row |
| Q2 | "extrapolating" banner | `ui/overlays.rs` `show` (comment at the end) |
| Q3 | per-atom uncertainty tint | `viewer.rs` `distance_overlay` (named in the `ui/overlays.rs` seam comment) |
| Q4 | container walls + size | `ui/menus.rs` Scene ▸ Container |
| Q5 | live-cloud refresh | `ui/menus.rs` View ▸ Density |
| Q6 | bond lines and molecule names | `ui/menus.rs` View ▸ Display aids; `ui/dock/selection.rs` Bonds |
| Q7 | composition badge | `ui/menus.rs` View ▸ Display aids; `ui/dock/selection.rs` Bonds |
| Q8 | relax (FIRE) | `ui/menus.rs` Physics ▸ Estimated solver |
