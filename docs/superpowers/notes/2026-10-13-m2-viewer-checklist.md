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
| Coincident placement → red "coincide" error | Code reading | worker path: `sim::provider_failure_pauses_with_message` (M4; was `coincident_nuclei_report_an_error_not_a_crash`); UI path is the same as the charge errors above |
| Apply grid 33/0.3 recreates the volume | Code reading | upload recreates on `volume_dims != [n;3]` (also hit by H₂ 49³ → H₂O 57³); worker: `sim::grid_change_resizes_the_density` |
| Close → window disappears at once, process exits | In-window (scripted) | `WM_CLOSE` during an H₂O solve → exit in ~140 ms, exit code 0 (commit 90d5a76) |

Known gap: the H₂O interactive solve at 57³ takes ~2 min (116 s, 13 SCF steps), not the plan's
"~10 s". This is solver performance, not the viewer; the UI stays responsive because the solve
runs on the worker thread.

## M4 Task 6 — dynamics in the viewer

Checked on 2026-10-13 against `m4-dynamics` (RX 7900 XTX, 240 Hz, GPU backend unless stated),
with stderr redirected to a log: the viewer prints one line per second while playing (time, T,
E_total, E_conserved, drift, fs/s, ms/step, SCF steps, the 0–1 distance), a frame-interval line
every 5 s while playing, and a line for every snapshot that leaves the worker idle. The window was
driven by posted Space keys and `WM_CLOSE`. Posted mouse input does not reach egui (the real cursor
was outside the window), so panel clicks were not scripted. The user was also using the window
during two sessions; the items they exercised are marked "In-window (user)", with the evidence
taken from that session's log. "Headless" means a viewer test that drives the same `Viewer`
code the window runs: `cargo test -p app --release`.

| Item | Status | Evidence |
|---|---|---|
| Starts on H₂ at R = 1.6 Bohr, paused | In-window | log: E_KS −0.980640 Ha, 8 SCF steps, 196–250 ms cold (GPU), 6.4 s (CPU). The capture shows "paused — converged", the Play button and the empty sparkline placeholder |
| Play / Pause with Space | In-window | the play log starts at once; Pause logs an idle snapshot within one step. Once, a posted Space did not take and the next one did |
| Vibration | In-window | d(0,1) runs 1.600 ↔ 2.017 Bohr. Minima at t = 0, 20.75 and 52.0 fs give a period of **10.4 fs**. T runs 0 ↔ 890 K. E_total stays in −0.980640 … −0.980669 Ha, a Verlet swing of 3e-5 Ha (≈ (ω dt)²/4 of the 4e-3 Ha exchanged). After 100 steps the drift is +8e-5 … −7e-7 Ha/ps, ~1e-5 typical |
| Speed of the run | In-window | 2.2–3.5 fs/s at 77–120 ms per step (4–7 SCF steps, ~240–280 readbacks), 49³ |
| Frame time while playing | In-window | interval p50 **4.2 ms**, p95 5.1–5.8 ms, max 6–38 ms per 5 s window. The 30–38 ms maxima fall in the first window after Play |
| Readouts and sparkline | In-window | captures show T, KE, E_KS, E total, E conserved (Ha and eV), drift, SCF steps / readbacks / ms per step, fs/s and steps per tick, and the E_total/T sparkline in anti-phase. The drift turned red in the runaway below. Amber was seen in the log only (+1.07e-4 Ha/ps), not in a capture |
| Play / Select / orbit with the mouse | In-window (user) | the user's session: Play, the Select tool, a camera orbit and a zoom |
| Drag while playing = spring | In-window (user) + test | user log: E_conserved separates from E_total by the spring energy while the drag is held, and T spikes to 2.4e4 K. `drag_while_playing_uses_the_spring_not_teleport`: max \|Δr\| per snapshot 0.021 Bohr, no jump |
| Select + drag while paused = teleport | In-window (user) | user log: an idle snapshot after the move, d(0,1) 4.19 → 2.45 Bohr, 11 SCF steps |
| Provider failure pauses with a message | In-window + test | GPU: "SCF did not converge after 200 iterations (Δρ = 1.9e-4)" at d = 2.61 Bohr during the user's run paused it with the red line; Space resumed. `provider_failure_pauses_with_message`: "nuclei 0 and 1 coincide" on both backends |
| CPU backend playing | In-window (`--backend cpu`) | 2.4–3.6 s per step, 0.06–0.11 fs/s. The trajectory matches the GPU run at step 21 (d(0,1) 2.0167 Bohr, T 0.2 K) and step 25 (1.9838 Bohr, 220.8 K). Frame interval p95 4.4–4.9 ms during CPU steps |
| Close during a CPU MD step | In-window (scripted `WM_CLOSE`) | the process exited in 142 ms |
| CPU ↔ GPU radio | Headless | `switching_backend_respawns_the_worker_with_the_current_scene` (the panel's settings are re-sent; the run restarts paused) |
| Pin, Rewind, thermostat, gravity preset | Headless | `panel_controls_reach_the_driver`: the pinned nucleus is bit-still over 5 steps; Rewind flips every velocity exactly; switching the thermostat on at rest gives T = 300 K and the Maxwell–Boltzmann note; g = 1.0836e-22 Bohr/a.u.² reaches the driver |
| Place / Delete while playing | Headless + code | `mirror_follows_the_dynamics_and_keeps_unconfirmed_edits` (an addition while playing shows at once and survives stale snapshots); driver: `add_and_remove_while_playing` |
| Drag target marker, pin squares, box-face warning | Code + unit test | `Viewer::distance_overlay`; `box_warning_flags_nuclei_near_or_beyond_the_face` |
| Save / Load with velocities | Unit test | `converts_to_and_from_the_dynamics_scene_with_velocities` (vx vy vz written only when non-zero) |
| Display aids during play | In-window | the captures show the cell box, the floor grid and the axes. The distance guides were not exercised (they need a mouse drag) |

**Runaway seen in the user's session (not a viewer bug; reported to the controller).** A drag
while playing gave the molecule centre-of-mass momentum, and with COM correction on nothing
stopped it at the box face. E_total rose from −0.934 to +0.16 Ha while E_conserved stayed flat at
−0.932: the energy entered through the COM correction's booked work. T reached 1.2e5 K, and the
nuclei left the 12 Bohr box (d(0,1) up to 35 Bohr; SCF steps up to 137). The panel now warns when
a nucleus is within 1.5 Bohr of the box face, or outside the box.

### Task 6 fix round 1 (R12, R13, review I1/I2/m1)

None of the items below was exercised in a window. They need a mouse, and posted mouse input
does not reach egui; re-check them by hand.

| Item | Status | Evidence |
|---|---|---|
| A nucleus closer than 3h to a face, or outside the cell, pauses with "left the simulation cell" (R12) | Unit + worker test | provider `nuclei_near_or_outside_the_cell_face_are_refused` (CPU and GPU, refused before any solve); app `a_nucleus_at_the_cell_face_pauses_with_a_message` |
| Drag spring force capped at 0.1 Ha/Bohr, work booked exactly (R12) | Unit test | `drag_force_is_capped_and_its_work_booked`: the first step is exactly F_max; E_total peaks at the capped potential, 0.18994 vs 0.19000 Ha (uncapped would be 1 Ha); \|ΔE_conserved\| ≤ 1.3e-3 Ha |
| "work done by tools" readout (E_total − E_conserved) | Code | `ui::dynamics_readouts`, with a tooltip on the drag-speed trade-off |
| A paused teleport warm-starts (R13) | Worker test | `moving_a_nucleus_warm_starts` (h2 49³, 0.02 Bohr): warm 381 readbacks / 7 steps vs cold 477 / 8, ratio 0.80 < 0.85; ΔE 1e-8 Ha |
| The drag mode follows Play/Pause at every motion event, never a teleport while playing (I1) | Unit test | `tools::drag_mode_follows_the_play_state_at_each_motion`. A select-click without motion no longer tethers the nucleus (review m2) |
| Drag targets, teleports and placements clamped into the cell minus 3h and ½h (I2) | Unit test | `tools::clamp_keeps_positions_inside_the_validation_margin`. The marker draws the clamped target |
| Focus loss or the cursor leaving the window while a button is held counts as a release (m1) | Code | `App::release_all` (`WindowEvent::Focused(false)`, `CursorLeft`) |
| Atom card (M4+, user request): selecting a nucleus shows a top-right Bohr-style schematic with Z protons, N neutrons (most abundant isotope), electron shells, and the not-to-scale / shared-electrons caveat | Code reading + unit tests (`atom_card::tests`) | `crates/app/src/atom_card.rs`, hooked in `Viewer::run_ui` |
