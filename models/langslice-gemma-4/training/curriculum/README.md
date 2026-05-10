# Dynamic Difficulty Curriculum (DORMANT)

A WeightedRandomSampler that biases training toward sections in
coordinate-bin regions where the model is currently weak. Built and tested,
but not actively used yet — defaults are uniform weights, behaviorally
identical to no curriculum.

For top-level orientation, see
[`../README.md`](../README.md).

## Status

| Component | Status |
|---|---|
| Bin computation (`bins.py`) | ✓ Built, unit-tested |
| `WeightedRowDataset` (`sampler.py`) | ✓ Built, KS-test verified |
| `CurriculumGRPOTrainer` / `CurriculumSFTTrainer` subclasses | ✓ Built, drop-in for default trainers |
| `CurriculumLogger` (per-section reward log) | ✓ Built |
| `compute_weights` (`weights.py`) | ✓ Built (per-bin MAE → adjusted weights with EMA + cap + floor) |
| Wiring into `train_grpo.py` / `train_sft.py` | ✓ via `--curriculum-weights` flag |
| Wiring into `iterate.py` | ✓ via `--curriculum-weights-dir` flag |
| Per-rollout logging from RLVR | ✓ Reward-callback wrapper appends to `curriculum_log.jsonl` |
| **Activated in any actual run** | ✗ **No. Defaults are uniform weights.** |

## What it does

Each section in the RLVR allocation is binned by its position-as-fraction-of-
plane-extent (5 quintiles per plane, using slicebench's `_bin_index`). After
each round's slicebench eval, per-bin MAE drives a weight update for the
next round's sampler. Bins where the model is weak get higher sample weight.

The weight formula is `w_bin ∝ (mae_bin / baseline_mae)^alpha`, with:
- 0.5-EMA smoothing across rounds (stable updates)
- 3× max change cap per round (prevents oscillation)
- 0.1× floor (no bin gets fully starved)

## Why it's dormant

We need a working baseline pipeline before tuning the sampler. Curriculum
optimization on top of a broken or unstable training loop just adds noise.
Once expert iteration is producing measurably-improving checkpoints
across rounds, turning on curriculum should give an additional ~5-15%
accuracy boost on weak coord bins (especially anterior coronal — q1 is at
60% accuracy on slicebench).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Lazy re-exports so importing the package doesn't pull TRL/torch |
| `bins.py` | `compute_section_bins(allocation) → dict[section_id, bin_idx]`. Imports `_bin_index` from `slicebench.score`. |
| `sampler.py` | `WeightedRowDataset(RowDataset)` + `CurriculumGRPOTrainer`/`CurriculumSFTTrainer` subclasses overriding `_get_train_dataloader()`. |
| `weights.py` | `read_per_bin_mae`, `compute_weights`, `update_weighted_dataset`. |
| `log.py` | Append-only JSONL per-section reward logger. |
| `cli.py` | Standalone `python -m curriculum.cli` weight-update tool. |

Tests in `tests/test_curriculum_*.py`. **35 tests pass**.

## How to activate

For expert iteration, add these flags to `iterate.py`:

```
--curriculum-weights-dir out/expert_iteration/run_<id>/curriculum_weights `
--curriculum-alpha 1.0 `
--curriculum-max-weight-change 3.0 `
--curriculum-floor-fraction 0.1 `
--curriculum-smoothing 0.5 `
```

iterate.py will:
1. Round 0: sample uniformly (no prior weights file).
2. After round 0 eval: compute per-bin MAE from
   `<run_dir>/round_0_slicebench/summary.json`. Write
   `<curriculum_weights_dir>/round_0_weights.json`.
3. Round 1: read `round_0_weights.json`, bias prompt sampling toward weak
   bins, also pass to SFT retrain via `--curriculum-weights`.
4. After round 1 eval: write `round_1_weights.json` (smoothed against
   round 0). Repeat.

For RLVR, add to `python -m langslice_rlvr ...`:
```
--curriculum-weights <path-to-weights.json>
```

The trainer will detect `_weights` on the dataset and switch to
`CurriculumGRPOTrainer` / `CurriculumSFTTrainer` automatically.

## When to activate

After:
1. Expert iteration has run 2-3 successful rounds with stable per-round
   improvement (or at least non-regression).
2. We've confirmed which coord bins are persistently weak via slicebench
   `per_coord_bin` breakdowns.
3. We're ready to spend an extra round to validate the sampler works.

DON'T activate before then. Curriculum on a regressing pipeline
just compounds the regression by oversampling whatever bin the model is
currently failing at.

## Synth data hookup (Phase D, not built)

When the synthetic data pipeline lands, synth slices land in
`data/manifest/shards/<plane>/synthetic.jsonl` + allocations in
`data/manifest/allocations/<plane>/rlvr.jsonl`. The bin computation
auto-picks them up by their `position_mm`, so they slot into whichever
quintile they belong to. The sampler picks them up at whatever weight
their bin has. **No new code needed for synth integration** — the
abstractions handle it.
