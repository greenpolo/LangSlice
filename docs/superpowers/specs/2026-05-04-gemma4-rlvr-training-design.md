---
title: Gemma 4 E4B - RLVR Training Design
date: 2026-05-04
scope: Multi-turn RLVR for the langslice-gemma-4 estimation agent. SFT data design is in 2026-04-25-gemma4-sft-data-design.md.
status: design v2
---

# Gemma 4 E4B - RLVR Training Design

## 1. Goal

Reinforce the post-SFT Gemma 4 E4B's actual agent-loop policy on AP / ML / DV estimation tasks, using verifiable rewards on the final coordinate(s). Output is a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate`), running offline on a single 5090.

Two estimation kinds are in scope:
- **Single-slice** - one tissue image, one coordinate.
- **Multi-slice / group** - N tissue images from one brain at known interval, N coordinates in order.

Out of scope: image-gen registration review, thinking-mode training, and bbox grounding.

## 2. Stack

- **Model loader:** Unsloth `FastVisionModel.from_pretrained(..., max_seq_length=..., fast_inference=False)`.
- **PEFT:** Phase A can start from either a full post-SFT model directory or an SFT LoRA adapter directory. Adapter directories are detected via `adapter_config.json`; RLVR loads the recorded base model and attaches the SFT adapter with `PeftModel.from_pretrained(..., is_trainable=True)`. Full model/checkpoint paths still get a fresh RLVR LoRA via `FastVisionModel.get_peft_model(...)`. Phase B resumes the saved Phase A LoRA by loading the requested base/post-SFT path and attaching `PeftModel.from_pretrained(..., is_trainable=True)`.
- **RL trainer:** TRL `GRPOTrainer` with `environment_factory=LangSliceEstimateEnv`.
- **Reference example:** `huggingface/trl/examples/scripts/openenv/carla_vlm_gemma.py`.
- **Hard version pins:** see `requirements-rlvr.txt`.

Verified API facts as of 2026-05-04:
- TRL main removes `max_prompt_length`; configs keep `max_completion_length` and rely on dataset-side prompt budgeting.
- TRL PR #5323/#5448 allow multimodal tool responses from `environment_factory`.
- TRL PR #5521 drops overlong tool results rather than truncating them; it does not add image-boundary-aware truncation.
- TRL PR #5390 defines `GRPOConfig.stop_tool_names`; LangSlice uses `["submit_estimate", "submit_group_estimate"]`.
- Unsloth Gemma 4 RL guidance uses `fast_inference=False`; Unsloth vision loading returns `(model, processor)`.

## 3. File Layout

```text
models/langslice-gemma-4/training/
  rlvr/
    __init__.py
    env.py              - LangSliceEstimateEnv
    rewards.py          - gated closeness reward
    dataset.py          - HF dataset assembly and subject holdout
    train_grpo.py       - training driver
    atlas_grid.py       - pre-rendered atlas slice cache
  configs/
    grpo_default.toml   - default hyperparameters
    grpo_pilot.toml     - Phase A single-slice-only run
    grpo_phase_b.toml   - Phase B mixed run, resumes Phase A LoRA
requirements-rlvr.txt   - exact version pins
langslice_rlvr/          - repo-root python -m launcher shim
src/langslice_rlvr/      - installed console-script launcher shim
```

## 4. Environment

`LangSliceEstimateEnv` exposes exactly three sync public methods to TRL. Sync is the only intentional divergence from production ADK tools.

Production-compatible tool shapes:

```python
def fetch_atlas(positions_mm: list[float], tool_context=None) -> dict: ...
def submit_estimate(position_mm: float, reasoning: str, tool_context=None) -> dict: ...
def submit_group_estimate(positions_mm: list[float], reasoning: str, tool_context=None) -> dict: ...
```

`fetch_atlas` behavior:
- Clamp each position to `valid_range_mm`.
- Dedupe positions within `0.02 mm`.
- Cap each call at 8 positions.
- Return dict responses with image-before-text content blocks.
- Reject further calls after submit with `{"status": "error", "error": "ALREADY_DONE", ...}`.

`submit_estimate` / `submit_group_estimate` behavior:
- Validate task kind.
- Parse final positions.
- Store hidden rollout state for rewards.
- Set `done=True`.
- Return production-shaped dict responses.

Hidden ground truth is never exposed through a public method.

## 5. Atlas Grid

At training startup, pre-render atlas reference slices every `0.05 mm` for every `(atlas_name, plane)` pair used by train or eval rows. The high grid index is clamped inward to the last valid atlas coordinate so `get_reference_slice` is never called one bucket outside the atlas.

`fetch_atlas` then performs a bounded cache lookup rather than slicing on every tool call.

## 6. Rewards

There is exactly one reward function: a normalized, truncated Gaussian on final-coordinate closeness. No format reward, structure reward, malformed-tool penalty, submit-count bonus, or turn-count shaping is used.

The error is normalized by the active plane's valid coordinate span so the same raw-millimeter error is judged more strictly on shorter axes and more leniently on longer axes:

Single-slice:

```python
axis_span_mm = valid_range_mm[1] - valid_range_mm[0]
err_frac = abs(predicted_position_mm - ground_truth_mm) / axis_span_mm

if err_frac >= cutoff_frac:
    reward = 0.0
else:
    raw = exp(-0.5 * (err_frac / sigma_frac) ** 2)
    floor = exp(-0.5 * (cutoff_frac / sigma_frac) ** 2)
    reward = (raw - floor) / (1.0 - floor)
```

Defaults:
- `cutoff_frac = 0.10`
- `sigma_frac = 0.035`

- Exact hit: `1.0`.
- Error at or beyond 10% of the plane axis span: `0.0`.
- Errors inside the cutoff receive a smooth bell-shaped score rescaled to `[0, 1]`.

Group:
- Compute the same per-slice reward for each submitted coordinate.
- Return the arithmetic mean across slices.

Failure modes:
- No submit, malformed submit, wrong submit kind, or wrong number of group positions returns `0.0`.
- There is no additional penalty.

## 7. Dataset And Holdout

Each row contains:

```python
{
    "prompt": list[ChatMessage],
    "image" or "images": PIL or list[PIL],
    "atlas_name": str,
    "plane": str,
    "valid_range_mm": tuple[float, float],
    "ground_truth_positions_mm": tuple[float, ...],
    "kind": "single" | "group",
    "subject_id": str,
}
```

Source rows come from `references/TestImages/M*/ground_truth.json`.

Subject holdout:
- Split by subject id before groups are built.
- Default deterministic split: sorted subject ids, every 5th subject goes to eval (`eval_holdout_every = 5`, about 20%).
- No subject may appear in both train and eval.
- `eval_holdout_every <= 0` disables holdout for explicit smoke/debug runs only.

Mix:
- Phase A (`grpo_pilot.toml`): `single_fraction = 1.0`, single-slice only.
- Phase B (`grpo_phase_b.toml`): `single_fraction = 0.5`, 50/50 single/group.

## 8. Training Driver

Canonical launch from the repo root:

```bash
python -m langslice_rlvr \
  --config models/langslice-gemma-4/training/configs/grpo_pilot.toml \
  --sft-model out/sft/gemma4-e4b-langslice \
  --output-dir out/rlvr/phase_a \
  --test-images-root references/TestImages

python -m langslice_rlvr \
  --config models/langslice-gemma-4/training/configs/grpo_phase_b.toml \
  --sft-model out/sft/gemma4-e4b-langslice \
  --resume-from-adapter out/rlvr/phase_a \
  --output-dir out/rlvr/phase_b \
  --test-images-root references/TestImages
```

Core trainer shape:

```python
training_args = GRPOConfig(
    chat_template_kwargs={"enable_thinking": False},
    max_completion_length=2048,
    max_tool_calling_iterations=20,
    stop_tool_names=["submit_estimate", "submit_group_estimate"],
    importance_sampling_level="sequence",
    loss_type="dr_grpo",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_funcs=[position_reward],
    args=training_args,
    environment_factory=lambda: LangSliceEstimateEnv(atlas_grid=shared_atlas_grid),
)
```

Loader contract:
- If `--sft-model` contains `adapter_config.json`, it is treated as an SFT LoRA adapter. The loader reads `base_model_name_or_path`, loads that base model, then attaches the SFT adapter as trainable weights.
- If `--sft-model` does not contain `adapter_config.json`, it is treated as a directly loadable model/checkpoint and wrapped with a fresh RLVR LoRA for Phase A.
- If `--resume-from-adapter` is supplied, that adapter is attached as the trainable adapter after the base/post-SFT model path is loaded.

## 9. Phased Rollout

1. **Smoke:** 10 frozen rollouts, no optimizer step. Verify env tool parsing, image injection, `tool_mask`, reward shape, and context length.
2. **Phase A single-slice pilot:** `grpo_pilot.toml`, `single_fraction=1.0`, 100-300 steps. Gate: held-out subject MAE improves and no-submit rate stays low.
3. **Phase B mixed pilot:** `grpo_phase_b.toml`, `single_fraction=0.5`, resume Phase A LoRA with `--resume-from-adapter`. Gate: group MAE improves and single-slice MAE does not regress.
4. **Scale:** increase `num_generations` after reward curves stabilize. Consider LoRA rank 32 later.

## 10. Reuse Pointers

- `src/langslice_harness/atlas/` - atlas loading and slice extraction.
- `src/langslice_harness/harness/estimation/tools.py` - production tool signatures and return shapes.
- `src/langslice_harness/harness/estimation/prompts.py` - system prompt builders.
- `references/TestImages/M*/ground_truth.json` - RLVR ground-truth source.

## 11. Verification

Required local checks for this scaffold:

```bash
python -m pytest tests/test_rlvr_env.py tests/test_rlvr_rewards.py
python -m ruff check models/langslice-gemma-4/training/rlvr tests/test_rlvr_env.py tests/test_rlvr_rewards.py
```

Unit coverage:
- Env: tool surface, production-shaped signatures, dict responses, clamp/dedupe/cap, done-fence, hidden ground-truth privacy.
- Rewards: exact, axis-normalized bell score, cutoff edge, beyond-cutoff, malformed submit, no submit, mixed group mean.
- Dataset: deterministic subject holdout, zero train/eval overlap, expected sizes.
- Atlas grid: high-end boundary clamp.
- Driver: `stop_tool_names` config forwarding, repo-root launcher, `max_seq_length` model load, Phase B adapter resume.

Broader checks before a real run:
- `python -m pytest`
- `python -m ruff check .`
- `python -m basedpyright`
- `python -m langslice_harness version`
- `langslice version`

## 12. Risks

- TRL `environment_factory` and multimodal tool responses are recent. Pin and smoke-test before long runs.
- Verified TRL PR #5390 defines `stop_tool_names`, but if the installed TRL build rejects that field, use a TRL build containing PR #5390 before training.
- Unsloth PyPI metadata may lag the TRL features needed for this scaffold. Treat `requirements-rlvr.txt` as the source of the intended pins and verify resolver output before a long run.
- Image context accumulation can become large after repeated `fetch_atlas` calls. Use `max_tool_calling_iterations`, `max_completion_length`, and the submit stop tools as guardrails.

## 13. Cleanup After RLVR

After both phases land, wire the trained LoRA adapter into `models/langslice-gemma-4/inference/predict.py` and delete stale triplet-era scaffolding that is no longer part of the pipeline.
