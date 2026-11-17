# Live knowledge graph during training — short design

**Status:** approved by the user 2026-11-16 ("Lets do it"). Small change.

## Intent
The user wants the Knowledge panel's pair graph to update live while a training pass runs. Today it redraws only after a refit (`KnowledgeKey` keys on the model generation, and the map counts fitted samples only), so it looks frozen for the whole pass.

## Design (agreed)
- **Live rebuild:** while a sampling pass, a teach scan or any other source adds kept samples, the knowledge map is rebuilt at most every 2 s (the existing `VIEWS_MIN_INTERVAL` cadence and backstop), and whenever the count of kept-but-unfitted samples changes.
- **A third bar segment, "new, waiting for refit":** samples kept since the last refit, stacked on top of each bin's fitted bar and drawn as a lighter outline (not filled). Legend: taught · thin · new (waiting for refit).
- **Honest colours:** a bin's taught/thin class still comes from fitted samples only. New samples never turn a bin green until they are in the fit. After the refit, the outlined segment becomes part of the solid bar.
- **Pair card text:** "Taught 1.2–3.5 Bohr (57 samples) · +12 new since the last refit" when there are any. The "Still missing" line stays fitted-only.
- **Molecules trained:** the list shows the same "+k new" where relevant.
- **Display only:** no change to training, fitting, trust or the stored samples. Cost is milliseconds per rebuild.

## Approaches
1. **Count pending samples in the map (chosen):** extend `KnowledgeMap` with per-bin pending counts read from the store's unfitted tail. The key gains the pending count.
2. **Refit more often during a pass (rejected):** it changes training cost and behaviour for a display feature.
3. **Draw raw sample ticks without binning (rejected):** inconsistent with the bin classes and the "Still missing" text.

## Constraints
- Rule 5: only `providers` knows the store; `sim::KnowledgeMap` gains plain data (per-bin `pending` counts).
- Don't use the samples' energies or forces, only their geometry (which pair distances or arrangements they cover), exactly as the fitted counts are built.
- Forgotten, quarantined and box-excluded samples are never counted as pending.
- egui: any new widgets go through `ui::keyed` where ids could shift. Screenshot `--ui` keys are unchanged.

## Left to the builder
The exact outline styling (follow `theme`), and where the pending tail is read in `SurrogateKs::knowledge_map`.
