# langslice_rlvr/ — PARKED

> Moved from `src/langslice_rlvr/` on 2026-05-11 as part of the Phase 1
> restructure of the iSFT speed upgrade.

This package held the **multi-turn TRL GRPO environment factory** for vision-language
RL on Gemma 4. It was superseded on **2026-05-10** when the multi-turn TRL
approach was abandoned (too slow, degraded quality) in favor of the
single-turn RL pipeline at [`../single_turn_rl/`](../single_turn_rl/).

The code remains on disk because:
1. The dataset loading helpers (`load_rlvr_allocation`, `atlas_grid`)
   are still imported by the active iSFT pipeline.
2. The TRL environment-factory wiring is a working reference if the
   multi-turn approach is ever revisited.

For top-level orientation see [`../README.md`](../README.md).

## What's still consumed

| Module | Consumer | Status |
|---|---|---|
| `langslice_rlvr.dataset.load_rlvr_allocation` | `iSFT/iterate.py`, `iSFT/synth_corpus.py` | **Active** — RLVR allocation loader |
| `langslice_rlvr.atlas_grid.build_atlas_grid` | `sft/train_sft.py`, eval callbacks | **Active** — atlas grid lookups |
| `langslice_rlvr.dataset.RowDataset` | `adaptive.curriculum.sampler.WeightedRowDataset` parent class | **Active** |
| Multi-turn TRL trainer scaffolding (`__main__.py` etc.) | Nothing | **Parked** |

## Why it's parked, not deleted

Per project memory `project_gemma_rlvr_pipeline` (superseded 2026-05-10):
multi-turn TRL GRPO scaffolding was found too slow and quality-degrading.
The console-script entry was removed from `pyproject.toml` in commit
`b8e8508` so `pip install -e .` no longer wires `langslice-rlvr` as a CLI.
The repo-root `langslice_rlvr/` shim was also deleted in the same commit;
this is the canonical home now.

## Don't import this for new RL work

For new RL work, use `single_turn_rl/` instead. The single-turn pipeline
is byte-clean, doesn't carry the multi-turn TRL env-factory baggage, and
has the active 77-test suite plus the `--curriculum-weights` and
`--adaptive-accept` integrations on the iSFT side.
