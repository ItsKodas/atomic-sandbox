# Three small queued items (2026-10-20) — how to test

Branch `small-items-speed-eigensolve-vram`, one commit per item. Build: `cargo build -p app --release`.

## 1. Speed cap 10 000 steps per tick (`sim::dynamics::MAX_SPEED`)

- Toolbar ▸ Speed: the slider and the typed field now go to 10 000 (they were 1000). Typing 50000 clamps to 10 000.
- Estimated mode on H₂ (Model tab ▸ Start sampling, then Play): at 10 000 the run is unpaced ("as fast as possible") and each tick stops at its 16 ms budget; at 1000 it is now *paced* — 1000 steps per 16 ms ≈ 15 600 fs/s — where it used to be the unpaced top. dt never changes (rule 8).
- Automated: `cargo test -p sim --release` (the clamp test) and `cargo test -p validate --release --test dynamics`; the worker's "as fast as possible" rate test now uses `MAX_SPEED`.

## 2. The display eigensolve in the Activity tab

- Start on H₂ (GPU), open View ▸ Density ▸ show ▸ Orbitals (or the Orbitals tab) with unoccupied orbitals > 0, paused. The left dock's Activity tab shows a **Computing orbitals** job row ("4 unoccupied orbitals per spin … display only", elapsed time, "can't be cancelled" with the reason) while the eigensolve runs — ≈ 1.7 s for H₂ at 81³, 9–10 s for H₂O at 95³.
- When it ends, the **Solve log** gets an **orbitals** row: origin "orbitals", "4 unoccupied per spin", outcome "shown" (or amber "stopped" if a command interrupted it), and its wall time in the solve column. The new **orbitals** chip hides them. The counters do not move (not a teaching solve).
- Headless: `target\release\app.exe --screenshot shot.png --preset h2 --ui "show=orbitals;unocc=4;tab=activity"` — the shot waits for the payload, so the log row should be in the picture.
- Automated: `ui::tests::the_display_eigensolve_has_a_job_row_and_log_rows`, `sim` `activity::tests::the_display_eigensolve_is_reported_and_taken_once`, `progress::tests::the_probe_token_moves_with_work_not_with_time`.

## 3. GPU memory budget from the card

- Budget order: `ATOMIC_SANDBOX_VRAM_GB` → half the card's largest device-local heap (Vulkan, `gpu::GpuContext::device_local_memory_bytes`) → 12 GB.
- The RX 7900 XTX reports 25 753 026 560 bytes (24 GiB): budget **12.9 GB** (it was 12.0). `cargo test -p gpu --release --test context_timing -- --nocapture` prints the number.
- To see a refusal: Scene ▸ Quality ▸ Custom 0.08 with eight Ne in a 16 Bohr box, or set `ATOMIC_SANDBOX_VRAM_GB=1` and pick Accurate on H₂O. The message ends "over the 12.9 GB budget (50 % of this GPU's 25.8 GB; ATOMIC_SANDBOX_VRAM_GB overrides it)" — or "(set by ATOMIC_SANDBOX_VRAM_GB)" / "(the default: this GPU did not report its memory; …)". A worker-side refusal (e.g. a `--scene` file over the budget) says the same, with "the teacher's half of …" in estimated mode.
- Automated: `quality::tests::the_vram_budget_follows_the_variable_then_the_card_then_the_default`, `ui::tests::a_change_over_the_vram_budget_is_refused`.
