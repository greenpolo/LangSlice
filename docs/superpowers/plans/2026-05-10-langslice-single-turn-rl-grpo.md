# LangSlice Single-Turn GRPO Handoff for Claude

> **For agentic workers:** Read `AGENTS.md` first. Do not edit training-data shards,
> overrides, or allocations unless the user explicitly assigns the matching data
> role. This handoff is for the Gemma 4 E4B training project under
> `models/langslice-gemma-4/`.

**Goal:** Add a single-turn RL path that can improve LangSlice production
single-slice position estimation without reviving the expensive multi-turn RLVR
tool loop.

**Recommendation:** Implement single-turn GRPO as a new lane beside the parked
multi-turn RLVR trainer. Start with terminal-state final-answer RL and fixed-slate
final-answer RL. Do not train the whole production agent loop online until these
single-turn tasks show a pass@k-to-pass@1 improvement and transfer on SliceBench.

**Current date/context:** 2026-05-10. Hackathon v1 scope is single-slice agent
traces only.

---

## Sources To Keep Open

- Unsloth Sudoku Gemma 4 GRPO notebook:
  <https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb>
- Unsloth advanced RL docs:
  <https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation>
- Unsloth Gemma 4 fine-tuning guide:
  <https://unsloth.ai/docs/models/gemma-4/train>
- LangSlice training orientation:
  `models/langslice-gemma-4/training/README.md`
- Parked multi-turn RLVR trainer:
  `models/langslice-gemma-4/training/rlvr/README.md`

---

## Bottom Line

The user is right that LangSlice has an unusually good verifier: final coordinate
error against ground truth is cheap, deterministic, and continuous. That should
make RL worthwhile.

The failed run does not prove "RL is bad here." It shows that the current RL
formulation is too expensive and unstable:

- The current trainer is single-slice at the dataset level, but still multi-turn
  at the rollout level.
- Each rollout has tool-call interruptions, atlas image fetches, long completions,
  and production-style turn control.
- The 300-step GRPO run had flat reward and KL drift, then regressed badly.
- GPU utilization was low because Python/tool work dominated the loop.

Single-turn GRPO should be used to train the decision we actually care about:
given a production-equivalent visual state, output the final coordinate. The
multi-turn agent can still be used to generate states and to evaluate transfer.

---

## Why RL Is Still The Right Next Step

The key empirical fact is that standard SFT already made good answers sampleable.
The model produced estimates within about 10% error across 4 generations. That
is exactly the regime where RLVR should help: the capability is already in the
sampling distribution, and RL should increase the probability that the good
answer appears at generation 1.

So the failed RLVR run should not be interpreted as "LangSlice was not ready for
RL." It should be interpreted as "the current RL episode shape made the clean
reward hard to optimize."

The current multi-turn setup asks one final coordinate reward to assign credit
through landmark prose, broad fetch choices, atlas image tool results, narrow
fetch choices, submit timing, tool-call formatting, and long completions. Once
the policy drifts away from the SFT tool protocol, failures compound: missed
submit calls, extra turns, context growth, malformed tool use, fallback midpoint
behavior, and rising KL without coordinate improvement.

The practical conclusion is:

- SFT already taught enough of the agent loop.
- RLVR should not relearn the whole agent loop.
- RLVR should optimize short final-coordinate decisions.
- Production can keep the multi-turn loop as the evidence-gathering mechanism.

The next RL objective should look like:

```text
query image + fetched atlas evidence -> {"position_mm": 4.37}
```

and not like:

```text
query image -> landmarks -> fetch_atlas -> compare -> fetch_atlas -> submit_estimate
```

This is the conceptual reason to implement single-turn RL before returning to
online multi-turn tool RL.

---

## What Unsloth Is Actually Recommending

The Sudoku notebook is the relevant pattern:

- Build one prompt/task.
- Sample multiple completions per prompt with GRPO.
- Score each completion using deterministic reward functions.
- Keep the environment outside the model and make the reward cheap.
- Watch `reward`, `reward_std`, completion length, and KL during training.

Important differences from LangSlice:

- Sudoku emits Python code and executes it. LangSlice should emit strict JSON or
  a tool-call-shaped final answer.
- Sudoku has reward-hacking checks such as module/import restrictions. LangSlice
  analogues are no GT leakage, no section IDs in prompts, no GT-centered atlas
  slates except for an explicit oracle diagnostic, and strict parse/range checks.
- Sudoku uses a small `num_generations=2` in the notebook because it targets a
  free T4 example. For LangSlice, use 8 as the starting point because our existing
  RLVR notes already found `num_generations=8` more efficient than 4.

From the advanced RL docs:

- `num_generations` must be greater than 2 for GRPO to work well.
- Use high enough sampling temperature for diversity. Unsloth suggests around
  `temperature=1.0`; our current `0.9` is close.
- For sequence-level rewards, `importance_sampling_level="sequence"` is a stable
  choice. Our current GRPO config already does this.
- `loss_type="dapo"` is now the documented default; `dr_grpo` is also reasonable
  and already used locally.
- `scale_rewards="batch"` is supported, but Dr. GRPO-style no-scaling can avoid
  difficulty bias. Because our reward is already normalized by atlas axis span,
  compare `batch` vs `none` in a short smoke.
- Be cautious with `mask_truncated_completions`: the docs note KL/NaN issues.
  Single-turn LangSlice outputs should be short enough that truncation is rare,
  so start with `mask_truncated_completions=false`.
- Higher `beta` constrains policy drift. Our failed run had KL growth, so a small
  nonzero beta is worth testing if memory allows a reference model.

From the Gemma 4 guide:

- For Gemma 4 GRPO, `fast_inference=False` is required. Do not try to use vLLM
  for GRPO training.
- Use `FastVisionModel` for multimodal Gemma 4.
- Keep `finetune_vision_layers=false` first; fine-tune language/attention/MLP.
- Use normal chat roles and keep image content before text content.

---

## Current LangSlice Code Context

Current best model:

- `models/langslice-gemma-4/training/README.md` says the best checkpoint is
  `out/sft/docker-sft-1011-merged-bf16`.
- SFT base SliceBench tiny num_gens=4: all-samples MAE 2.88 mm, best-of-4 MAE
  1.15 mm, failure 3.5%.
- The RLVR run `out/rlvr/stage1-n8-overnight` regressed to 15.26 mm small-bench
  MAE with 55% failure.

Parked RLVR components to reuse carefully:

- `models/langslice-gemma-4/training/rlvr/train_grpo.py`
  - Loads `FastVisionModel`, forces `fast_inference=False`, builds `GRPOTrainer`.
  - Builds the atlas grid before training.
  - Attaches SFT adapters as trainable PEFT adapters.
- `models/langslice-gemma-4/training/rlvr/dataset.py`
  - Loads RLVR allocations and shard metadata.
  - Uses production `build_single_slice_prompt`.
  - Preserves Gemma 4 image-before-text ordering.
  - Uses SFT-matched image preprocessing and deterministic CLAHE assignment.
- `models/langslice-gemma-4/training/rlvr/rewards.py`
  - Has the useful `normalized_bell_reward(error_mm, axis_span_mm, cutoff_frac,
    sigma_frac)`.
- `models/langslice-gemma-4/training/rlvr/atlas_grid.py`
  - Pre-renders atlas slices on a 0.05 mm grid.
- `models/langslice-gemma-4/training/configs/grpo_pilot.toml`
  - Current useful settings: `num_generations=8`,
    `importance_sampling_level="sequence"`, `loss_type="dr_grpo"`,
    `temperature=0.9`, `top_p=0.95`.
  - Current risky setting for single-turn: `max_completion_length=3072` is far
    too long; `mask_truncated_completions=true` should be disabled initially.

Production transfer constraints:

- Production prompts in `src/langslice_harness/harness/estimation/prompts.py`
  expect broad fetch, compare, recover, narrow, and submit behavior.
- Production validators in `src/langslice_harness/estimation/_tool_logic.py`
  require broad and narrow sweep behavior before accepting submit calls unless
  the run is near the iteration limit.
- Production `fetch_atlas` can return a target-plus-atlas comparison grid.
- The current RLVR env returns atlas slices through a training-time tool loop.
  That is the slow part we should bypass for GRPO.

---

## Proposed Training Lanes

### Lane A: Terminal-State Final-Answer GRPO

Train on the state right before production should submit.

Prompt content:

- Query image.
- The same atlas images or comparison grids that the production agent already
  fetched during a broad/narrow search.
- Short instruction: output only `{"position_mm": number}`.

Reward:

- Parse JSON.
- Reject missing, nonnumeric, NaN, or out-of-range coordinates.
- Score with `normalized_bell_reward`.

Why this transfers:

- In production, after `fetch_atlas` calls, the model reaches the same kind of
  state and must choose the final coordinate.
- This trains the final submit decision without making GRPO learn the whole tool
  policy.

How to build states:

- Preferred: derive from existing SFT traces and expert-iteration traces by
  cutting each trace immediately before `submit_estimate`.
- Fallback: generate scripted terminal states with deterministic broad/narrow
  atlas panels. Use the SFT model's own estimate or a coarse nearest-neighbor
  policy to choose the narrow center. Do not center panels on GT for real
  training.

### Lane B: Fixed-Slate Final-Answer GRPO

Train a one-call estimator where the prompt contains the query image plus a
fixed deterministic atlas slate.

Prompt content:

- Query image.
- A broad atlas slate, for example 7 or 9 evenly spaced positions across the
  valid plane range.
- Optionally one deterministic local slate centered on a non-GT source, such as
  the current SFT model's greedy estimate.
- Output only `{"position_mm": number}`.

Reward:

- Same as Lane A.

Why this transfers:

- Production can prefetch the same deterministic slate before calling the model.
- If this beats the multi-turn path, it can become a new fast production mode.
- If it does not beat production, it still gives a cheap RL diagnostic.

Critical leakage rule:

- A GT-centered local slate is allowed only as an oracle upper-bound experiment.
  It must never be used for train/eval claims.

### Lane C: Next-Action Single-Turn GRPO

Train a one-step policy: given query image plus current fetched state, output the
next action JSON.

Output schema:

```json
{"action": "fetch_atlas", "positions_mm": [1.0, 3.0, 5.0]}
```

or

```json
{"action": "submit_estimate", "position_mm": 4.35}
```

Reward:

- For submit: coordinate reward.
- For fetch: shaped reward for valid range, non-duplicates, broad coverage if no
  broad sweep exists, tight bracketing near the best available candidate if a
  broad sweep exists, and mild cost penalty for excessive fetches.

Why this is later:

- It is more production-faithful, but reward design is less clean.
- Start with Lanes A/B, where the verifier is exactly the final coordinate.

---

## Recommended Implementation Shape

Create a new package:

```text
models/langslice-gemma-4/training/single_turn_rl/
  __init__.py
  README.md
  dataset.py
  prompts.py
  rewards.py
  train_grpo.py
  eval_single_turn.py
```

Create configs:

```text
models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml
models/langslice-gemma-4/training/configs/grpo_single_turn_slate.toml
```

Do not modify the parked multi-turn trainer except to import shared helpers if
needed. Keeping this separate makes it easy to compare against the failed RLVR
path and to delete if it underperforms.

Reuse:

- `rlvr.dataset.preprocess_query_image`
- `rlvr.dataset.load_rlvr_allocation`
- `rlvr.dataset._atlas_in_plane_long_edge` if kept importable, or move a public
  helper into a shared module.
- `rlvr.rewards.normalized_bell_reward`
- `rlvr.atlas_grid.build_atlas_grid` or production `_build_atlas_grid` depending
  on whether the prompt sends individual images or a labeled comparison grid.

Prefer production visual format for transfer:

- If production is usually `send_individually=false`, train on the target-plus-
  atlas grid image generated by `src/langslice_harness/estimation/_tool_logic.py`.
- If production uses individual atlas images, train on individual images in the
  same order, with image before caption text.

---

## Starting Hyperparameters

Use the SFT checkpoint as the initialization. Do not start from the failed RLVR
adapter.

Suggested initial single-turn config:

```toml
[grpo]
chat_template_kwargs = { enable_thinking = false }
learning_rate = 1e-6
num_generations = 8
generation_batch_size = 8
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
max_completion_length = 128
dataloader_num_workers = 4
dataloader_pin_memory = true
max_grad_norm = 0.1
importance_sampling_level = "sequence"
loss_type = "dapo"  # use "dr_grpo" if installed TRL drops this key
mask_truncated_completions = false
scale_rewards = "batch"
temperature = 1.0
top_p = 1.0
optim = "adamw_8bit"
log_completions = true
logging_steps = 1
save_steps = 25
max_steps = 100
report_to = "trackio"
load_in_4bit = true
max_seq_length = 4096

[lora]
finetune_vision_layers = false
finetune_language_layers = true
finetune_attention_modules = true
finetune_mlp_modules = true
r = 16
lora_alpha = 32
use_gradient_checkpointing = "unsloth"

[reward]
cutoff_frac = 0.15
sigma_frac = 0.05
format_penalty = -1.0
out_of_range_reward = 0.0
```

Run a small grid of smokes before spending an overnight run:

- `scale_rewards="batch"` vs `"none"`.
- `learning_rate=1e-6` vs `3e-6`.
- `num_generations=8` vs `16` if memory allows.
- Optional small `beta` if KL drifts again and memory permits reference-model
  loading.

---

## Required Diagnostics Before Real Training

Do this before any >100-step GRPO run:

1. Build a held-out single-turn eval set for Lane A or Lane B.
2. Run the current SFT model with `num_generations=1`, `4`, `8`, and optionally
   `16`.
3. Measure greedy MAE, best-of-N MAE, parse failure rate, out-of-range rate,
   reward mean, and reward std.
4. Proceed only if best-of-N is meaningfully better than greedy. RL is most
   likely to help when there is a clear pass@k-to-pass@1 gap.
5. Abort or redesign the prompt if reward std is near zero for most groups.

Interpretation:

- If best-of-8 is much better than greedy, GRPO can plausibly compress sampling
  ability into pass@1 behavior.
- If best-of-8 is not better, the prompt/state lacks the visual evidence needed
  for the model to solve the task, and RL will probably drift or overfit.

---

## Training Stop Rules

Do not repeat the 300-step failure mode.

Stop a run early if any of these hold after 25-50 steps:

- Reward mean is flat and reward std is low.
- KL grows monotonically without eval improvement.
- Completion length rises instead of staying near the JSON answer length.
- Parse failures increase.
- Held-out single-turn MAE worsens versus the SFT initialization.

A checkpoint is useful only if it passes both gates:

- Single-turn held-out eval improves pass@1 MAE or failure rate.
- Production SliceBench eval does not regress, ideally improves num_gen1 while
  preserving the existing best-of-4 ceiling.

Reference baselines from `training/README.md`:

- SFT base tiny all-samples MAE: 2.88 mm.
- SFT base tiny best-of-4 MAE: 1.15 mm.
- SFT base tiny failure rate: 3.5%.

---

## Evaluation Plan

Single-turn eval:

- Lane A terminal-state eval on held-out traces/states.
- Lane B fixed-slate eval on held-out sections.
- Report greedy and best-of-N.
- Report per-plane metrics. Coronal dominates existing data; do not hide
  sagittal/horizontal regressions in the aggregate.

Production eval:

- Merge or load the RL adapter the same way existing SFT/RLVR eval does.
- Run SliceBench tiny first, then small if tiny improves.
- Compare:
  - Original production multi-turn agent with SFT.
  - Original production multi-turn agent with RL adapter.
  - Fixed-slate production mode with RL adapter, if Lane B is implemented.

Success criteria:

- No increase in parse/fallback failures.
- Greedy num_gen1 improves, because production usually needs one answer.
- Best-of-4 does not degrade below the SFT ceiling.

---

## Notes On Atlas Embedding Caching

Do not make atlas embedding precomputation the first milestone for this RL lane.

It may become useful because fixed-slate/terminal-state training resends many of
the same atlas images. However, the main uncertainty is not image-encoder
throughput. It is whether the single-turn RL objective produces a checkpoint
that transfers to production. Keep the vision tower frozen, prove the training
objective first, and only then revisit embedding-cache splice work with a
bit-exact correctness gate.

---

## Concrete First Task For Claude

Implement Lane A first.

Expected output:

- A new `single_turn_rl` package.
- A terminal-state dataset builder that can consume existing SFT or
  expert-iteration traces and produce GRPO rows.
- A strict JSON final-answer prompt.
- A reward function that parses JSON and calls `normalized_bell_reward`.
- A trainer that mirrors `rlvr/train_grpo.py` but does not use
  `environment_factory`.
- A smoke config with `max_completion_length=128` and
  `mask_truncated_completions=false`.
- Unit tests for parsing, reward scoring, out-of-range handling, and dataset row
  image-before-text ordering.

Implementation guardrails:

- Do not edit manifest shards, overrides, or allocations.
- Do not use GT-centered atlas panels for training.
- Do not train on model rationales initially. Output JSON only.
- Do not change production inference until single-turn eval shows improvement.
- Do not reuse `out/rlvr/stage1-n8-overnight`.

Suggested initial command shape after implementation:

```powershell
python -m single_turn_rl.train_grpo `
  --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml `
  --sft-model out/sft/docker-sft-1011-trimmed-noeval `
  --terminal-states out/single_turn_rl/terminal_states.jsonl `
  --output-dir out/rlvr_single_turn/terminal_smoke
```

Then immediately run:

```powershell
python -m single_turn_rl.eval_single_turn `
  --model out/rlvr_single_turn/terminal_smoke `
  --eval-states out/single_turn_rl/terminal_eval.jsonl `
  --num-generations 1 4 8
```

The exact module path may need `PYTHONPATH=models/langslice-gemma-4/training`
or a repo-root shim, following the existing `langslice_rlvr` pattern.

---

## If Lane A Stalls

Move to Lane B before returning to multi-turn RL.

Lane B has the clearest relationship to Unsloth's Sudoku example: one prompt,
fixed evidence, deterministic final reward. It also creates a possible new
production estimator that avoids tool loops entirely:

1. Build deterministic atlas slates.
2. Train `query + slate -> JSON coordinate`.
3. Evaluate as a one-call production path.
4. If it works, use the multi-turn agent only as a fallback for hard cases.

Only attempt Lane C after A/B show that single-turn coordinate RL improves the
model.
