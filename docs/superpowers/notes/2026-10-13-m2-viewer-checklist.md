# M2 viewer — manual checklist status

The viewer's manual checklist is Task 7, Step 5 of `docs/superpowers/plans/2026-10-13-m2-viewer.md`.
This note records how far each item was verified when M2 landed. Until M5 adds the
input-replay harness, these items have no automated coverage beyond the unit tests named here.
Re-run the unverified ones by hand after UI or render changes: `cargo run -p app --release`.

"In-window" means it was seen in a running window: a capture or the stderr log, from the
implementer's own instance or the user's live session. "Code reading" means the logic was
checked (and unit-tested where noted), but nobody exercised it in a window.

| Item | Status | Evidence |
|---|---|---|
| Window opens (RX 7900 XTX, Vulkan, `Rgba8UnormSrgb` surface) | In-window | log + window capture |
| H₂ appears with its density; snapshot uploads with right dims | In-window | 49³ × 0.25 Bohr, ρmax 0.2231, E −0.962716 Ha, 8 SCF steps, ~5.9 s; capture shows density + spheres |
| Resize | In-window (user) | maximised to 2560×1369; correct rendering afterwards, no panic |
| Minimise | Code reading | `SurfaceState::resize` ignores 0 sizes; `redraw` returns early at 0×0; `about_to_wait` stops requesting redraws while minimised (polls instead) |
| Preset switching (He, H₂O) | In-window (user) | He → E −2.138685; H₂O → spheres update at once, spinner shown, old density kept meanwhile |
| Isosurface mode | In-window (user) | capture |
| Solver error → red message, old picture stays | In-window (user) | charge > protons on H₂ → red "exceeds the 2 protons" line, previous density still drawn |
| Gamma slider → volume updates | In-window (user) | capture at gamma 0.55 |
| Orbit / pan / zoom | Code reading | `render::Camera::orbit/pan/zoom`; `Viewer::zoom` floors the distance at 2.0 |
| Place adds a nucleus on the focus plane, density updates | Code reading | `tools::place_point` unit-tested |
| Display aids: cell box, floor grid (xz, +y up) with axes, placement-plane grid in Place mode, 1 Bohr / 1 Å spacing | Code reading + headless | `render::lines` pass tested headless (lit pixels, occluded by spheres, no depth write); `overlay::build` unit-tested |
| Distance guides: faint lines + Bohr/Å labels from a dragged nucleus (or the Place cursor) to nuclei within 6 Bohr | Code reading | `tools::distance_guides` / `tools::project` unit-tested; overlay in `Viewer::distance_overlay` (egui foreground layer) |
| Select + drag, Delete key | Code reading | `tools::pick` / `tools::drag_point` unit-tested; drag sends `MoveNucleus` per motion event, the worker coalesces (`sim::queued_commands_are_coalesced_into_one_solve`) |
| Coincident placement → red "coincide" error | Code reading | worker path: `sim::coincident_nuclei_report_an_error_not_a_crash`; UI path is the same as the charge errors above |
| Apply grid 33/0.3 recreates the volume | Code reading | upload recreates on `volume_dims != [n;3]` (also hit by H₂ 49³ → H₂O 57³); worker: `sim::grid_change_resizes_the_density` |
| Close → window disappears at once, process exits | In-window (scripted) | `WM_CLOSE` during an H₂O solve → exit in ~140 ms, exit code 0 (commit 90d5a76) |

Known gap: the H₂O interactive solve at 57³ takes ~2 min (116 s, 13 SCF steps), not the plan's
"~10 s". This is solver performance, not the viewer; the UI stays responsive because the solve
runs on the worker thread.
