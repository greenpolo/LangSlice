# GRPO RLVR Trainer (PARKED)

This directory holds the multi-turn agent GRPO RLVR trainer (Group Relative
Policy Optimization with verifiable rewards). It is **parked** as of 2026-05-09
in favor of expert-iteration SFT (see
[`../iSFT/README.md`](../iSFT/README.md)).
Code is preserved for future use and reference.

## Why parked

A 300-step GRPO run from the SFT base (`stage1-n8-overnight`) produced a
**worse** model than the SFT base on slicebench:

| Metric | SFT base | RLVR after 300 steps |
|---|---|---|
| Slicebench tiny MAE (num_gens=4) | 2.88 mm | n/a (didn't finish) |
| Slicebench small MAE (num_gens=1) | n/a | **15.26 mm** |
| Failure rate | 3.5% | **55%** (rolled out longer than context window, fell back to midpoint) |

The training-time signal predicted this exactly: reward stayed flat across
all 300 steps (mean 0.24 throughout), while KL divergence grew 17× from
start to end. Classic policy-wander signature — the model drifted away
from the SFT init without finding a better policy.

The resulting checkpoint is at `out/rlvr/stage1-n8-overnight/` (LoRA adapter)
but should NOT be used as a base for future training. The SFT-merged-bf16
remains the best checkpoint we have.

## What we learned (read before un-parking)

1. **vLLM does NOT support Gemma 4 GRPO.** Per [Unsloth docs](https://unsloth.ai/docs/models/gemma-4/train),
   `fast_inference=False` is mandatory. This means rollouts run sequentially
   through Unsloth-native generation, ~190s/step at num_gens=4 + tool calls.
2. **The bottleneck is the multi-turn agent loop, not the GPU.** GPU util
   stays at ~25% during training because the model has to stop mid-rollout
   for tool calls (atlas image fetches), CPU does Python work, GPU idles.
   The 5090 is loafing.
3. **Reward shape matters but doesn't save you.** We calibrated cutoff_frac
   and sigma_frac based on the actual model's slicebench distribution
   (best-of-4 lands at 5.5% plane error). Reward signal was alive (mean 0.24,
   std 0.4) but didn't translate to MAE improvement.
4. **`num_generations=8` works better than `4`** with Unsloth-native batching:
   we measured ~50% per-rollout cost reduction at N=8 vs N=4.
5. **Multi-token prediction (MTP) was released for Gemma 4** but it's
   inference-only, text-only, no help for vision RL training.

## Files

| File | Purpose |
|---|---|
| `train_grpo.py` | Main GRPO driver. Uses Unsloth FastVisionModel + TRL GRPOTrainer. |
| `dataset.py` | RLVR allocation loader, `RowDataset`, prompt building. |
| `env.py` | `LangSliceEstimateEnv` — multi-turn agent env with `fetch_atlas` tool. |
| `rewards.py` | Truncated-Gaussian position reward function. |
| `atlas_grid.py` | Pre-rendered atlas reference slices on a 0.05mm grid. |

## How to run (if you ever un-park)

```powershell
# Edit configs/grpo_pilot.toml first, especially num_generations + reward shape.
.\scripts\docker-training\rlvr.ps1 -RunName "stageN-name" -ReportTo "none"
```

Default config (`configs/grpo_pilot.toml`):
- num_generations=8, generation_batch_size=8
- max_completion_length=3072, max_seq_length=6144
- max_steps=300
- cutoff_frac=0.15, sigma_frac=0.05 (slicebench-calibrated)

The trainer's `--sft-model` defaults to `out/sft/docker-sft-1011-trimmed-noeval`
(SFT LoRA adapter). It loads the base + that adapter, then attaches a fresh
GRPO LoRA on top.

Saves adapter to `out/rlvr/<run_name>/`. Use the wave-2 merge helper in
`../iSFT/vllm_lifecycle.py` to merge for vLLM serving.

## Why we'd ever un-park

GRPO might still work IF:
1. We get better-shaped rewards (curriculum-aware? finer than truncated Gaussian?)
2. We ship the dynamic-difficulty sampler (curriculum/) so hard slices get
   more rollouts and easy ones get fewer
3. We get vLLM-RL support for Gemma 4 (currently blocked upstream)
4. We have a stronger SFT base — RL works best when starting from a model
   that's already mostly right

For now, expert iteration gives more bang for the buck on our hackathon
timeline. Keep the GRPO code maintained for future stages.

## Known pre-existing test failures

`tests/test_rlvr_env.py` has 5 failing tests as of 2026-05-09:
- 1 from config drift (`grpo_pilot.toml` has `num_generations=8`, test asserts 4)
- 4 from `MagicMock(spec=GRPOConfig).__mro__` issues in `_filter_grpo_config_for_installed_trl`

These are not blocking. Documented for awareness.
