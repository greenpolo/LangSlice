---
title: Gemma 4 E4B — RLVR Training Design
date: 2026-05-04
scope: Multi-turn RLVR for the langslice-gemma-4 estimation agent. SFT data design is in 2026-04-25-gemma4-sft-data-design.md.
status: design v1
---

# Gemma 4 E4B — RLVR Training Design

## 1. Goal

Reinforce the post-SFT Gemma 4 E4B's actual agent-loop policy on AP / ML / DV estimation tasks, using verifiable rewards on the final coordinate(s). Output is a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate`), running offline on a single 5090.

Two estimation kinds in scope:
- **Single-slice** — one tissue image → one coordinate.
- **Multi-slice / group** — N tissue images (sliced from one brain at known interval) → N coordinates in order.

Out of scope: image-gen registration review (separate brainstorm), thinking-mode training (off initially), bbox grounding (covered by SFT only).

## 2. Stack (locked-in)

- **Model loader:** Unsloth `FastVisionModel.from_pretrained(..., fast_inference=False)`.
  - Why Unsloth: Gemma 4 patches (KL fix from PR #4934, `final_logit_softcapping` injection, KV-shared layer fixes), ~60% VRAM reduction, custom backward kernels.
  - Why `fast_inference=False`: Unsloth's published guidance for Gemma 4 RL; sidesteps the vLLM hybrid-attention performance issue (vLLM #38887). Unsloth-native inference, no Ray, no Docker.
- **PEFT:** `FastVisionModel.get_peft_model(finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing="unsloth")`. LoRA rank 16 first, scale to 32 after stable.
- **RL trainer:** TRL `GRPOTrainer` with `environment_factory=LangSliceEstimateEnv`. Multi-turn agent training, automatic `tool_mask`, multimodal tool responses (TRL PR #5323, April 2026).
- **Reference example:** `huggingface/trl/examples/scripts/openenv/carla_vlm_gemma.py` — Gemma 4 + multimodal tool responses + LoRA, almost line-for-line our pattern with a different env.
- **Hard version pins:** `transformers>=5.2.0`, `trl` ≥ the commit that includes PR #5448 (test + docs for multimodal tool responses, 2026-04-06) and PR #5521 (image-boundary-aware tool truncation, 2026-04-11). Pin these in a `requirements-rlvr.txt`.

## 3. Why this stack (not the alternatives)

We evaluated and ruled out:

- **OpenPipe ART** — text-only trajectories.
- **OpenRLHF 0.10** — Ray + vLLM mandatory; tight on a single 5090 without Unsloth's optimizations.
- **ms-swift v4.2** — requires ≥2 GPUs (one for rollout server).
- **verl-agent / prime-rl / Verifiers** — vLLM-rollout-based; designed for distributed.
- **GAD-cell/vlm-grpo** — single-turn only; needs a multi-turn extension we'd write ourselves.
- **Custom `_generate_and_score_completions` subclass** — was the leading plan before TRL's `environment_factory` was found. PR #5093 (Feb 2026) + PR #5323 (April 2026) make this unnecessary; the trainer handles per-rollout env instances and multimodal tool injection natively.

The TRL path keeps everything on a single 5090 with Unsloth's full benefits, no Ray / Docker / vLLM setup, and ~400 LOC of custom code that follows a published-and-tested example (`carla_vlm_gemma.py`).

## 4. File layout

```
models/langslice-gemma-4/training/
  rlvr/
    __init__.py
    env.py              — LangSliceEstimateEnv (~150 LOC)
    rewards.py          — reward functions (~100 LOC)
    dataset.py          — HF dataset assembly (~50 LOC)
    train_grpo.py       — driver, ports carla_vlm_gemma.py (~100 LOC)
    atlas_grid.py       — pre-rendered fine-grained atlas slice cache (~50 LOC)
  configs/
    grpo_default.toml   — hyperparameters
    grpo_pilot.toml     — phase 2 / 3 overrides
  notebooks/
    sft.ipynb           — Phase 1, vendored Gemma4_(E4B)-Vision.ipynb
    grpo.ipynb          — Phase 2+, thin wrapper around train_grpo.py
requirements-rlvr.txt   — version pins
```

## 5. LangSliceEstimateEnv (env.py)

A single class. Public methods (anything not starting with `_` and not named `reset`) are auto-exposed as tools to the model by `GRPOTrainer.environment_factory`.

```python
class LangSliceEstimateEnv:
    def __init__(self, atlas_grid: AtlasGrid):
        # Pre-rendered atlas slice cache, shared across rollouts.
        self._atlas_grid = atlas_grid
        self._reset_state()

    def reset(self, *, prompt_messages, image_or_images, atlas_name, plane,
              valid_range_mm, ground_truth_positions_mm, kind) -> None:
        # Stash hidden ground-truth (NOT exposed via any tool).
        # Configure for `kind` ∈ {"single", "group"}.
        # Return None — initial user prompt is already in train_dataset rows.
        ...

    # --- TOOLS (auto-exposed by environment_factory) ---

    def fetch_atlas(self, positions_mm: list[float]) -> list[dict]:
        """Render atlas slices at the requested positions."""
        # Clamp to valid range; dedupe within 0.02 mm; cap at 8 per call.
        # Return list of content blocks per position:
        #   [{"type":"image", "image": PIL_slice}, {"type":"text", "text":"Atlas at X.XX mm"}]
        ...

    def submit_estimate(self, position_mm: float, reasoning: str) -> str:
        """Final answer for single-slice tasks."""
        # Record; sets self._done = True; rejects if kind != "single".
        ...

    def submit_group_estimate(self, positions_mm: list[float], reasoning: str) -> str:
        """Final answer for group tasks."""
        # Record; sets self._done = True; rejects if kind != "group".
        ...
```

Reward functions read env state via the `environments` kwarg that `GRPOTrainer` passes to them.

### 5.1 fetch_atlas validation

- Clamp each position to `valid_range_mm` (clipping signaled in returned text).
- Dedupe positions within 0.02 mm (no point fetching the same slice twice).
- Cap at 8 positions per call. Excess positions dropped, signaled in returned text.
- All N (≤8) slices returned in a single tool response, image-before-text per Gemma 4's chat-template rule.

### 5.2 Atlas grid pre-render (atlas_grid.py)

A startup cost paid once per training run, not per rollout:

- For each (atlas, plane) pair in the training set, pre-render slices every 0.05 mm across the valid range.
- Store as PIL Images in a dict keyed by `(atlas_name, plane, round(position_mm * 20))`.
- `fetch_atlas` does dictionary lookup, not on-the-fly slicing.

Reuses existing slice extraction in `src/langslice_harness/atlas/` — does not reimplement.

## 6. Rewards (rewards.py)

Three additive components. All get `environments` (list of env instances per rollout) via kwargs from `GRPOTrainer`.

### 6.1 position_accuracy_reward (primary)

Single-slice:
- Tolerance: `tol = max(0.02 * (pos_hi - pos_lo), 0.20 mm)` — atlas-relative, floored at 0.20 mm.
- Tiered: `1.0` if `|err| ≤ tol`, `0.5` if `≤ 2*tol`, `0.25` if `≤ 4*tol`, `0` else.
- No submission → `0`.

Group:
- Per-slice: same tiered scoring as single.
- Aggregate: `0.5 * mean_per_slice_score + 0.5 * worst_per_slice_score` — penalizes outliers.
- Wrong number of submitted positions → `0`.

### 6.2 format_compliance_reward (small)

- +0.2 if exactly one submit call at end of trajectory.
- +0.2 if submit kind matches task kind (`submit_estimate` for single, `submit_group_estimate` for group).
- -0.2 per malformed tool call along the way.
- -0.5 if no submission of any kind.

### 6.3 length_budget_reward (small)

- 0 within budget (8 turns single, 12 turns group).
- -0.05 per excess turn until trajectory ends.

Process rewards (6.2 + 6.3) max ≈ 0.4. Accuracy reward dominates so the model can't win by "performing the ritual" while missing the coordinate.

## 7. Dataset (dataset.py)

HF `Dataset` rows:

```python
{
    "prompt": list[ChatMessage],   # image-before-text per Gemma 4 rule
    "image" or "images": PIL or list[PIL],
    "atlas_name": str,             # e.g. "allen_mouse_25um"
    "plane": str,                  # "coronal" / "sagittal" / "horizontal"
    "valid_range_mm": tuple[float, float],
    "ground_truth_positions_mm": list[float],   # length 1 for single, N for group
    "kind": str,                   # "single" | "group"
    "subject_id": str,             # for subject-level holdout
}
```

Sources:
- `references/TestImages/M0[1-9]/ground_truth.json` — known-label tissue images.
- SFT-pipeline holdout — tissue from datasets in the manifest with confirmed atlas-registration metadata. RLVR uses a strict subject-level holdout from the SFT and eval splits (defined in `eval/dataset_allocation.md`).

Mix: 70% single-slice, 30% group. Adjust after Phase 4 if group estimation lags.

System prompt comes from `src/langslice_harness/harness/estimation/prompts.py:build_single_slice_prompt` / `build_group_prompt` — used verbatim, no separate RL prompt.

## 8. Training driver (train_grpo.py)

Ports `examples/scripts/openenv/carla_vlm_gemma.py` to LangSlice. Key config:

```python
training_args = GRPOConfig(
    chat_template_kwargs={"enable_thinking": False},   # thinking off initially
    learning_rate=5e-6,
    num_generations=2,                                  # scale to 4 after stable
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=4096,
    max_completion_length=2048,
    max_tool_calling_iterations=20,                     # cap per rollout
    max_grad_norm=0.1,
    importance_sampling_level="sequence",               # GSPO
    loss_type="dr_grpo",
    optim="adamw_8bit",
    log_completions=True,
    logging_steps=2,
    save_steps=50,
    report_to="trackio",
)

trainer = GRPOTrainer(
    model=unsloth_loaded_model,                        # post-SFT checkpoint
    train_dataset=train_dataset,
    reward_funcs=[position_accuracy_reward, format_compliance_reward, length_budget_reward],
    peft_config=lora_config,
    args=training_args,
    environment_factory=lambda: LangSliceEstimateEnv(atlas_grid=shared_atlas_grid),
)
trainer.train()
```

## 9. Phased rollout

Total ≈ 5-7 working days for RLVR (after SFT). Hackathon deadline 2026-05-18.

1. **Smoke (~½ day).** 10 frozen rollouts (no optimizer step) on a tiny manifest. Verify env tool parsing, image injection, `tool_mask`, reward shapes, context length.
2. **Single-slice synthetic pilot (~1 day).** 100–300 steps on clean + augmented atlas targets only. Gate: reward curve trends up, MAE on held-out synthetic improves. If reward is flat or trajectories collapse to no-submit, debug before scaling.
3. **Single-slice mixed pilot (~1 day).** Add real histology with subject-level holdout. Gate: held-out MAE improves vs. SFT baseline, no-submit rate stays low.
4. **Group estimation (~1 day).** Flip 30% of dataset to `kind="group"`. Keep single-slice in the mix to avoid forgetting. Gate: group MAE improves, single-slice MAE doesn't regress.
5. **Scale (~1 day).** `num_generations=4` if reward curve is stable. Consider LoRA rank 32. Optional vision-projector unfreeze if visual matching is the failure mode.

Each gate is a kill switch — if a phase doesn't improve, stop and debug rather than push to the next.

## 10. Reuse pointers

Do not duplicate; depend on existing code.

- `models/langslice-gemma-4/data/synth_dataset.py` — `SynthIterator`, `render` for any synthetic queries injected into RLVR data.
- `models/langslice-gemma-4/data/augmentation/` — modality pipelines.
- `src/langslice_harness/atlas/` — slice extraction (used by `atlas_grid.py`).
- `src/langslice_harness/harness/estimation/tools.py` — match `fetch_atlas` / `submit_estimate` / `submit_group_estimate` signatures exactly so the trained adapter is wire-compatible with production at inference.
- `src/langslice_harness/harness/estimation/prompts.py` — system prompt builders, used verbatim.
- `references/TestImages/M0[1-9]/ground_truth.json` — RLVR ground-truth source.

## 11. Verification

1. **Env unit tests** (`tests/test_rlvr_env.py`):
   - `fetch_atlas` clamps, dedupes, caps at 8 positions, returns image+text content blocks.
   - `submit_estimate` rejects when `kind="group"` and vice versa.
   - Hidden ground-truth is not exposed by any public method.
2. **Reward unit tests** (`tests/test_rlvr_rewards.py`):
   - Canned trajectory with `|err|=0.1mm` → `position_accuracy_reward=1.0`.
   - Trajectory with no submission → `format_compliance_reward=-0.5`.
   - 12-turn single-slice trajectory → `length_budget_reward=-0.2`.
3. **SFT smoke** (notebook): 50 steps on 100-example subset; loss decreases; checkpoint saves and loads.
4. **RLVR smoke** (driver): 10 rollouts no optimizer step; verify all reward components log; verify tool-call iterations stay under cap.
5. **Phase 2 pilot run**: full 100-step run, reward curve, eval MAE on held-out synthetic.
6. **Inference smoke**: load post-RLVR adapter into `models/langslice-gemma-4/inference/predict.py`; run on `references/TestImages/M01/M01_001_001.tif`; |error| < 1.0 mm (loose, just confirms wire compatibility).
7. **Existing harness regression**: `python -m pytest tests/` clean.
8. **Type / lint**: `python -m ruff check models/langslice-gemma-4/training/`, `python -m basedpyright models/langslice-gemma-4/training/`.

## 12. Risks and follow-ups

- **TRL `environment_factory` is marked experimental** ("This feature is experimental and may change or be removed at any time"). Pin a known-good commit and vendor if upstream changes break us.
- **TRL multimodal tool responses are ~30 days old** (PR #5323 merged 2026-04-02). Edge cases possible — chat-template tool-token handling for Gemma 4 specifically may need patching. Smoke test catches this.
- **Unsloth-native inference is slower than vLLM** but the Gemma 4 vLLM hybrid-attention perf issue (vLLM #38887) makes Unsloth-native competitive in practice for our use case. Expect single-digit hours per ~500-step run.
- **Image context accumulation** by turn 8 can be 8-15 atlas images. May need `max_completion_length` tuning or per-turn image deduplication if we hit context limits.
- **Vision tower stays frozen** by default. If post-Phase-4 the failure mode is visual matching (not strategy), unfreeze the projector layer (`finetune_vision_layers=True` for projector only) before considering full vision-tower fine-tune.

## 13. Cleanup (after both phases land)

Delete stale scaffolding from the older triplet plan (already flagged in `models/langslice-gemma-4/README.md`):
- `models/langslice-gemma-4/data/build_triplets.py`
- `models/langslice-gemma-4/data/distill_cot.py`
- `models/langslice-gemma-4/data/generate_atlas_slices.py`
- `models/langslice-gemma-4/training/finetune.py`

After RLVR completes, wire `models/langslice-gemma-4/inference/predict.py` to load the trained LoRA adapter and drive the production agent loop.
