# GPU memory budget setting — design (short)

**Approved by the user 2026-11-21** ("yes build it").

## Why

On a GPU with 8 GB of memory, v0.2.0 paused Estimated mode at step 2 with "numerical failure in the solver: no gap witness within 3 extra levels … 3 is the most the 2.1 GB budget (the teacher's half of the 4.2 GB budget: 50 % of this GPU's 8.4 GB) fits".

The budget is fixed at 50 % of the card (`quality::ADAPTER_SHARE`). Estimated mode splits it between the teacher and the auditor. At the Accurate grid that leaves room for only 3 witness levels. The only way to change it today is the `ATOMIC_SANDBOX_VRAM_GB` environment variable, which users won't find.

## What

- **The control.** A "GPU memory" slider in **Scene ▸ Quality**, below the grid levels, also reachable with Ctrl K "gpu memory".
  - Range 25 %–90 % of the card's reported memory, default 50 % (today's behaviour).
  - The label shows the result, e.g. "GPU memory: 60 % of this GPU's 8.4 GB = 5.0 GB (teacher 2.5 GB + auditor 2.5 GB)". In GPU mode: "(one solver)". On the CPU backend: "not used by the CPU solver".
  - Above 70 % there is a warning line: the screen, the driver and other applications share the card.
  - If the card did not report its memory, the slider is disabled and explains why (the 12 GB default applies; set `ATOMIC_SANDBOX_VRAM_GB`).
- **Applies at once.** A change is a `ui::Action`, so `--record` sees it. The worker rebuilds its GPU solvers (teacher, auditor or the GPU backend's solver) with the new budget, the way a backend switch does. The scene is kept, paused, and re-solved, with no other state lost: the estimated model and its collection stay as they are.
- **Persisted** in the profile under a new key. Per `docs/dev-reference.md` "When to bump SCENE_FORMAT and PROFILE_FORMAT", a new key alone does not bump `PROFILE_FORMAT`.
  - **Amended 2026-11-21 (user report: the slider reset between runs):** saved per computer in `%LOCALAPPDATA%\atomic-sandbox\machine.settings` instead, never in a profile. A profile is shared between machines and builds, so a named profile's Load, Reset, or an older build rewriting `current.profile` could drop or move it. 0.2.1-beta.1's `gpu_memory` line is moved there once.
- **The environment variable wins.** When `ATOMIC_SANDBOX_VRAM_GB` is set, the slider is shown locked with "set by ATOMIC_SANDBOX_VRAM_GB = N GB".
- **Refusal wording.** `quality::VramBudget::why()` names the share ("60 % of this GPU's 8.4 GB, set in Scene ▸ Quality; ATOMIC_SANDBOX_VRAM_GB overrides it"). When a witness search fails at the memory cap (the cap set by the budget is below the hard cap of 32), the paused message adds: "raise GPU memory in Scene ▸ Quality, or use a coarser quality (Balanced)".
- **Every place that reads the budget** (`quality::vram_budget()` callers: the quality refusal, the box advice, the builder, segments, the worker) sees the new share at once, through the same global the adapter memory uses.

## Not doing

- Entering a value in GB: a percentage carries between machines.
- Changing the teacher/auditor 50/50 split.
- Choosing the default quality from the card's memory: a separate item, queued.

## Approaches considered

1. **A slider whose change rebuilds the solvers (chosen).** It takes effect immediately and uses the existing backend-switch path.
2. **A slider that applies at the next restart.** Simpler, but a confusing UX for a fix the user wants now.
3. **Only document the environment variable.** Users would never find it.

## Gates

- `cargo test -p app --release`, including the replays.
- `cargo test -p providers --release` if the refusal text in `ks.rs` changes.
- No physics changes.
- README (the Quality row, plus a line in the 8 GB note) and DEVLOG.
