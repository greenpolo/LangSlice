---
title: Gemma 4 E4B — SFT Training Code Design
date: 2026-05-05
scope: Supervised fine-tuning training code for the langslice-gemma-4 model. Consumes the SFT corpus produced by separate data sessions (single bucket — agent traces); produces a LoRA adapter that feeds the RLVR phase via `train_grpo.py --sft-model`.
status: design v1.3 — langslice-native trace format; renderer owns HF translation
---

# Gemma 4 E4B — SFT Training Code Design

## 1. Goal

Build the supervised fine-tuning code that takes a JSONL corpus of distilled agent-loop traces and produces a LoRA adapter the RLVR phase can load. Output is the post-SFT checkpoint that `models/langslice-gemma-4/training/rlvr/train_grpo.py` consumes via `--sft-model`.

The model is Gemma 4 E4B (5090 / 32 GB VRAM constraint per memory note `feedback_gemma_model_size_e4b`). Stack mirrors RLVR (`docs/superpowers/specs/2026-05-04-gemma4-rlvr-training-design.md` §2): Unsloth `FastVisionModel` 4-bit, LoRA on language layers + attention + MLP, vision tower frozen.

## 2. Scope

In scope:

- File layout under `models/langslice-gemma-4/training/sft/`.
- JSONL data contract (renderer reads, data sessions write).
- Hyperparameter config (`configs/sft_default.toml`).
- Train-time evaluation (held-out loss + agent-loop MAE).
- Verification (unit tests, smoke run, RLVR handoff).
- Cleanup of dead scaffolding flagged in the RLVR spec §13.

Out of scope:

- SFT data generation. Separate sessions; this spec only defines the consumer contract.
- SliceBench eval rig. Separate work; this spec only confirms the trained adapter loads into it.
- Vision-tower fine-tuning. Deferred per RLVR spec §12.
- Anatomy-grounding buckets (landmark listing, bbox grounding, multi-slice morphology). Cut from v1 due to schedule; see §3.

## 3. Bucket revision (since 2026-04-25 SFT data spec)

The 2026-04-25 SFT data spec (`docs/superpowers/specs/2026-04-25-gemma4-sft-data-design.md`) described five buckets — agent traces, landmark listing, bbox grounding, multi-slice morphology, programmatic skeletons. v1 collapses to **a single bucket: agent loop traces** distilled from Gemini 3.1 Pro running the production estimation agent loop. This teaches tool format and final-position output, which is the only training signal directly tied to the deployment task.

Anatomy-grounding buckets (landmark listing, bbox grounding, multi-slice morphology) and programmatic skeletons are deferred entirely. They remain options if v1 SFT under-performs post-RLVR — but the data-collection cost was judged not worth the schedule hit before the 2026-05-18 hackathon deadline.

This revision supersedes §5–§5.4 of the 2026-04-25 SFT data spec for the v1 implementation. The earlier spec's content remains relevant if scope is later expanded.

The JSONL row schema retains a `bucket` field set to `1` so the data sessions don't need to refactor if additional buckets return in v2. The trainer treats all rows uniformly in v1.

## 4. Architectural decisions

### 4.1 Vision tower frozen

Same as RLVR. Pretrained vision encoder handles general visual primitives (edges, contrast, anatomical structure) well enough; with the v1 corpus size we cannot meaningfully retrain it and risk catastrophic forgetting if we try. Trainable parameters are LoRA adapters on language layers, attention modules, and MLP modules. Aligns with RLVR config so SFT and RLVR adapters are interchangeable.

If post-RLVR the failure mode is visual matching itself (model picks atlas slices that don't look like the query), the cheap escalation is unfreezing only the projector layer, not the full vision tower (RLVR spec §12).

## 5. File layout

```
models/langslice-gemma-4/training/
  sft/
    __init__.py
    train_sft.py        — driver script (parallel to rlvr/train_grpo.py)
    dataset.py          — JSONL loader + subject-aware train/eval split
    render.py           — Example → Gemma 4 chat messages
    collate.py          — applies processor.apply_chat_template, builds labels with -100 outside assistant turns
  configs/
    sft_default.toml    — hyperparameters
  rlvr/                 — (existing, untouched)
requirements-rlvr.txt   — extended in place; no new requirements file
```

## 6. Data contract (JSONL)

Data sessions emit one JSONL file at a path supplied via `--dataset` on the CLI. One JSON object per line. Each row uses a **langslice-native trace format** that mirrors the production estimation agent loop's natural shape (`fetch_atlas` → tool result → ... → `submit_estimate`). The format is intentionally readable: opening the JSONL shows what the model did at each step, in the same vocabulary the production loop uses.

The renderer (`render.py`) translates each trace into HuggingFace chat-template format at training time. Data sessions never need to know about HF chat-template specifics, `tool_call_id` pairing, or content-block conventions — those concerns are owned by the trainer side.

### 6.1 Schema

```json
{
  "bucket": 1,
  "atlas_name": "allen_mouse_25um",
  "atlas_version": "CCFv3",
  "plane": "coronal",
  "subject_id": "<unique brain identifier for subject-level holdout>",
  "system_prompt_kind": "single_slice",
  "query_image_paths": ["queries/M03_001_001.png"],
  "user_prompt_text": "<exact user-message text Gemini saw at the start of the loop>",
  "trace": [
    {
      "tool_call": {"name": "fetch_atlas",
                    "args": {"positions_mm": [3.0, 5.0, 7.0]}},
      "tool_result": {
        "image_paths": ["atlas/allen_mouse_25um/coronal/3.00mm.png",
                        "atlas/allen_mouse_25um/coronal/5.00mm.png",
                        "atlas/allen_mouse_25um/coronal/7.00mm.png"],
        "text": "Atlas at 3.00 mm | 5.00 mm | 7.00 mm"
      }
    },
    {
      "tool_call": {"name": "fetch_atlas",
                    "args": {"positions_mm": [4.5, 5.0, 5.5]}},
      "tool_result": {
        "image_paths": ["..."],
        "text": "..."
      }
    },
    {
      "submit": {"name": "submit_estimate",
                 "args": {"position_mm": 5.2}}
    }
  ],
  "gemini_reasoning": "<optional, ignored unless include_rationale=true>"
}
```

Notes on the schema:

- **`trace`** is a flat sequence of steps. Each step is either:
  - A `tool_call` + `tool_result` pair (one round of tool use). The pairing is implicit — the `tool_result` belongs to the `tool_call` in the same step entry.
  - A terminal `submit` step. Must be the last entry in `trace`.
- **`tool_call.name`** is the production tool name (`fetch_atlas`). **`tool_call.args`** is the arg dict in production shape (`positions_mm: list[float]` etc.) — same as what `langslice_harness.harness.estimation.tools` defines.
- **`tool_result.image_paths`** are paths relative to the dataset JSONL's parent directory. **`tool_result.text`** is the human-readable string the production tool returns to the model (e.g. `"Atlas at 3.00 mm | 5.00 mm"`).
- **`submit.name`** is `submit_estimate` (single-slice) or `submit_group_estimate` (group). **`submit.args`** matches the production tool's arg dict: `{"position_mm": <float>}` for single, `{"positions_mm": [<float>, ...]}` for group.
- **`query_image_paths`** is a list (length 1 for single-slice, N for group). The user turn shows all of them in order.
- **`user_prompt_text`** captures the exact user-message text the production loop sent — same source the data session distilled from.
- **System prompt and tool-schemas are NOT stored in the JSONL.** The renderer pulls the system prompt from `src/langslice_harness/harness/estimation/prompts.py` (via `system_prompt_kind`) and the tool-schemas from `src/langslice_harness/harness/estimation/tools.py`. Production code stays the single source of truth; stale copies in distilled traces can't drift.
- **`bucket`** is fixed at `1` in v1, retained as a discriminator field so future buckets can be added without schema migration.

### 6.2 Validation

`dataset.py` validates each row on load:

- Required fields present per schema; `bucket == 1`.
- `subject_id` non-empty (subject-level holdout integrity).
- `system_prompt_kind ∈ {"single_slice", "group"}`.
- `query_image_paths` non-empty; for `single_slice` exactly length 1.
- `trace` non-empty; final entry is a `submit` step; all preceding entries are `tool_call` + `tool_result` pairs.
- Final `submit.name` matches `system_prompt_kind` (`submit_estimate` for single, `submit_group_estimate` for group).
- For group: `submit.args.positions_mm` length equals `query_image_paths` length.
- Every `image_paths` entry resolves on disk relative to the JSONL parent.

Malformed rows raise a clear error citing line number and the specific failure. No silent skipping.

## 7. Renderer + collator (render.py + collate.py)

The pipeline has two responsibilities, split across files because they operate at different stages:

- `render.py` operates on the in-memory `Example` (parsed JSONL row) → translates the langslice-native trace into HF chat-template format (`messages` + `tools`) with image paths hydrated to PIL images. Output is what `processor.apply_chat_template(...)` consumes.
- `collate.py` operates on a batch of rendered examples → calls the processor, then constructs the labels tensor with `-100` (ignore) on every token that is *not* an assistant response.

This split matters because the loss mask cannot be precomputed at the `messages` level: the processor expands every `image` content block into many image-placeholder tokens at the last second, and the mask must be computed *after* that expansion.

### 7.1 Renderer (render.py) — langslice-native → HF chat-template

Inputs: one parsed JSONL row in langslice-native format (§6.1).

Output: a dict ready for the processor — `{"messages": [...], "tools": [...], "metadata": {...}}` where `metadata` carries fields the collator may need (atlas version, subject id, optional assistant-turn span hints for the §7.3 fallback path).

Steps:

1. **System prompt.** Load from `src/langslice_harness/harness/estimation/prompts.py:build_single_slice_prompt` or `build_group_prompt` per `system_prompt_kind`. Emit `{"role": "system", "content": <prompt>}` as the first message.
2. **Tools schema.** Construct the HF `tools` list from `src/langslice_harness/harness/estimation/tools.py`. For `single_slice`: `[fetch_atlas, submit_estimate]`. For `group`: `[fetch_atlas, submit_group_estimate]`. Each tool entry follows the HF function-schema shape (`{"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON schema>}}`). Built once and threaded into `apply_chat_template(..., tools=tools)`.
3. **User turn.** Emit one `user` message containing every `query_image_paths` entry as `{"type": "image", "image": <PIL>}` content blocks (image-before-text per Gemma 4's chat-template rule), followed by `{"type": "text", "text": <user_prompt_text>}`.
4. **Trace translation.** For each `trace[i]` entry:
   - If it has `tool_call` + `tool_result`:
     - Generate a deterministic `tool_call_id` (e.g. `f"call_{i}"`).
     - Emit `{"role": "assistant", "tool_calls": [{"id": <id>, "type": "function", "function": {"name": <tool_call.name>, "arguments": json.dumps(<tool_call.args>)}}]}`.
     - Emit `{"role": "tool", "tool_call_id": <id>, "content": [<image blocks for tool_result.image_paths>, {"type": "text", "text": <tool_result.text>}]}`.
   - If it has `submit` (terminal step):
     - Generate `tool_call_id` (e.g. `"call_final"`).
     - Emit `{"role": "assistant", "tool_calls": [{"id": <id>, "type": "function", "function": {"name": <submit.name>, "arguments": json.dumps(<submit.args>)}}]}`.
     - No matching `tool` message — the trajectory ends here.
5. **Image hydration.** Resolve every `image_paths` entry to a `PIL.Image` instance, opened with the JSONL parent directory as the resolution root.
6. **Renderer-side validation.** Re-check the translated `messages`: every assistant `tool_calls[].id` (except the final submit) has a matching `tool` message; `arguments` strings are JSON-parseable; final submit's function name matches `system_prompt_kind`.

The renderer does *not* tokenize or apply the chat template — that happens in the collator. The renderer's job ends at producing the message list the processor knows how to consume.

### 7.2 Collator (collate.py)

Inputs: a batch (list) of rendered examples.

Output: the standard `(input_ids, attention_mask, pixel_values, labels, image_grid_thw, …)` tensor dict the model consumes, with `labels` masked correctly.

Steps:

1. For each example, call `processor.apply_chat_template(messages, tools=tools, chat_template_kwargs={"enable_thinking": False}, add_generation_prompt=False, tokenize=True, return_assistant_tokens_mask=True, return_dict=True, return_tensors="pt")`. The `return_assistant_tokens_mask=True` flag returns a per-token mask of which positions correspond to assistant-generated content (this is the standard HuggingFace mechanism, replacing the old text-only `train_on_responses_only` helper).
2. Pad / collate per-example tensors into the batch.
3. Construct `labels = input_ids.clone()`. Then set `labels[assistant_mask == 0] = -100` to ignore everything outside assistant turns. **This single step masks system / user / tool / image-placeholder tokens uniformly** — image tokens fall outside assistant turns by construction, so no special-casing is needed once we trust `return_assistant_tokens_mask`.
4. Sanity check (asserts during smoke run, deletable in prod): for each example, verify that no `labels[i] != -100` corresponds to an image-placeholder token ID. If the check fires, the chat template's assistant-mask logic is wrong for our trace shape and the collator falls back to a manual span-based mask reconstructed from the renderer's `metadata.assistant_turn_spans`.

### 7.3 Why the off-the-shelf path doesn't work

TRL's `assistant_only_loss=True` (the standard "train on responses only" flag) does not support VLM datasets — it's text-only. Unsloth's `train_on_responses_only` helper has the same limitation. The chat-template `return_assistant_tokens_mask=True` mechanism *does* work for VLMs as long as the template is well-behaved around image placeholders, which is what Gemma 4's published template aims for. The implementation plan's first verification step confirms this on the Gemma 4 E4B tokenizer before trusting it; the manual-span fallback in step 3 above exists for the case where it isn't.

## 8. Training driver (train_sft.py)

CLI:

```bash
python -m models.langslice-gemma-4.training.sft.train_sft \
    --config models/langslice-gemma-4/training/configs/sft_default.toml \
    --dataset models/langslice-gemma-4/data/sft_examples.jsonl \
    --output-dir out/sft/run0 \
    [--seed 0]
```

Flow:

1. Load TOML config; load JSONL dataset; subject-aware 90/10 split (no `subject_id` appears in both partitions).
2. Lazy-import `unsloth` + `trl` + `transformers` (so unit tests can import sibling modules without a runtime install).
3. `FastVisionModel.from_pretrained(base_model, load_in_4bit=True, max_seq_length=...)` from the `[sft]` table.
4. `FastVisionModel.get_peft_model(model, r=16, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing="unsloth")`. The returned object is already a `PeftModel` — do **not** also pass a `peft_config` to `SFTTrainer`. Passing both is a TRL hard error.
5. Wrap train/eval datasets in `RenderedDataset` (a thin `torch.utils.data.Dataset` shim that calls `render.py` per-row and caches results). The collator (`collate.py`) is supplied separately to the trainer.
6. Construct `SFTTrainer` (TRL) with:
   - `model` = the LoRA-wrapped `PeftModel` from step 4. **No `peft_config` argument.**
   - `processing_class` = `processor` returned by `FastVisionModel.from_pretrained`.
   - `args` = `SFTConfig(output_dir, num_train_epochs, learning_rate, …)` populated from `[sft]` table. **`assistant_only_loss=False`** (the helper does not support VLMs; we mask manually in the collator).
   - `data_collator` = the custom collator from `collate.py` (sets `labels = -100` outside assistant turns).
   - `train_dataset` / `eval_dataset` = the `RenderedDataset` partitions.
7. Register a `BaselineEvalCallback` that runs **once before training** with the base model only (no adapter active): produces `position_mae_mm` and `tool_call_parseability_rate` on `references/TestImages/M0[1-9]/`. This baseline goes into trackio so post-training numbers are anchored to "did SFT actually move the needle?" instead of an absolute number with no reference point.
8. Register an `AgentLoopEvalCallback` that fires every `agent_eval_steps`:
   - Activates the current adapter for inference (Unsloth `FastVisionModel.for_inference` or equivalent — verify exact API in implementation phase).
   - Runs the agent loop on `references/TestImages/M0[1-9]/`.
   - Logs `position_mae_mm`, `no_submit_rate`, `tool_call_parseability_rate`, `mean_trace_length` to trackio.
   - Restores training-mode after eval (`FastVisionModel.for_training`).
9. `trainer.train()`; `trainer.save_model(output_dir)`. Saves the LoRA adapter weights + tokenizer/processor config. The base model is *not* re-saved.

Logging: every step → trackio (loss, lr, gradient norm). Adapter saved to `--output-dir` on every `save_steps` interval.

## 9. Hyperparameter config (sft_default.toml)

```toml
[sft]
base_model = "unsloth/gemma-4-e4b-it"   # verify exact ID via Unsloth docs before commit
load_in_4bit = true
max_seq_length = 16384                  # multi-turn + multi-image traces blow past 4K easily
num_train_epochs = 3
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
weight_decay = 0.01
optim = "adamw_8bit"
chat_template_kwargs = { enable_thinking = false }
logging_steps = 5
eval_steps = 50
agent_eval_steps = 200
save_steps = 100
report_to = "trackio"
seed = 0

[lora]
r = 16
lora_alpha = 32
finetune_vision_layers = false
finetune_language_layers = true
finetune_attention_modules = true
finetune_mlp_modules = true
use_gradient_checkpointing = "unsloth"

[data]
holdout_fraction = 0.10
include_rationale = false
```

Notes on defaults:

- `num_train_epochs = 3` is generous for a small corpus. Drop to 1 if loss plateaus at epoch 1; bump to 5 if still falling at epoch 3.
- `learning_rate = 2e-4` is the standard QLoRA starting point. Lower to 1e-4 if loss is too noisy.
- `per_device_train_batch_size = 1` + `gradient_accumulation_steps = 8` → effective batch size 8. Higher per-device batch likely OOMs on 5090 with 4-bit E4B and image tokens.
- `max_seq_length = 16384` accounts for ~6–10 atlas-return turns × 3 images each at typical Gemma 4 image-token expansion rates, plus tool-call text. **Verify against Gemma 4 E4B's actual context window during implementation** — if the model maxes out at 8K, raise `agent_eval_steps` and reduce dataset diversity to fit; if it supports 32K+, raising further is cheap with gradient checkpointing.

## 10. Evaluation

### 10.1 Pre-training baseline (one-shot)

Before optimizer steps begin, run the agent-loop eval against the unmodified base Gemma 4 E4B (no adapter). Logs the same metrics §10.2 lists. This gives the post-training numbers a "did SFT actually help?" anchor — without it, an absolute MAE has no comparison point. Costs ~5–10 minutes once.

### 10.2 During training

Two cadences, both logged to trackio:

- **Every `eval_steps` (default 50): held-out loss** on the eval partition. Cheap, automatic. Confirms training is going somewhere.
- **Every `agent_eval_steps` (default 200): agent-loop run** on `references/TestImages/M0[1-9]/` ground-truth-labeled images. Slow (~5–10 minutes per pass). Logs:
  - `position_mae_mm` — primary metric (lower is better).
  - `tool_call_parseability_rate` — fraction of cases where every emitted tool call (including the final submit) parses as valid JSON with required arg keys present and types correct. **Critical pre-RLVR gate** (see §10.3).
  - `no_submit_rate` — fraction of test images where the model failed to call any submit at all.
  - `mean_trace_length` — average turns per example.

`position_mae_mm` trending down is the "is this getting better?" signal. Secondary signals are diagnostic — they explain why MAE is whatever it is when it's bad (no-submit, tool format collapse, flailing).

### 10.3 Pre-RLVR parseability gate

RLVR's reward (`docs/superpowers/specs/2026-05-04-gemma4-rlvr-training-design.md` §6) is dominated by the position-accuracy term, with only small format-compliance penalties. If post-SFT the model emits malformed tool calls most of the time, RLVR's accuracy reward never fires — RL devolves into mostly zero-reward exploration and won't converge.

Hard gate before invoking `train_grpo.py`: **`tool_call_parseability_rate >= 0.80`** on the held-out agent-loop eval. If post-training is below 0.80, the SFT corpus is too thin for tool format to land — remediations are to enlarge the trace corpus, add the deferred programmatic-skeletons bucket back in (cheap to generate, format-correct by construction per the 2026-04-25 SFT data spec §5 bucket 5), or to extend training. Do not move to RLVR until the gate passes.

### 10.4 Post-training quality gate

SliceBench (in development separately) — apples-to-apples against Flash and Pro baselines. If SliceBench MAE is meaningfully worse than Flash, debug before RLVR rather than letting RL chase a broken initialization (per SFT data spec §12).

## 11. Verification

- `tests/test_sft_render.py` — canned single-slice and group langslice-native traces → verify the renderer's translation to HF chat-template format: system prompt prepended from `prompts.py`, `tools` schema sourced from `tools.py`, image paths hydrated to PIL, `tool_call_id` pairing generated correctly between assistant and tool turns, JSON-stringified `arguments` round-trips, terminal submit's function name matches `system_prompt_kind`.
- `tests/test_sft_dataset.py` — tiny canned JSONL → verify schema validation errors are clear and line-attributed; verify subject-aware 90/10 split has no `subject_id` leakage.
- `tests/test_sft_collate.py` — fake batch of two examples through the actual processor → assert `labels` is `-100` on every system/user/tool/image-placeholder token and equals `input_ids` on every assistant token. Includes the §7.2 step-3 sanity check (no labels on image-placeholder token IDs).
- **Smoke run (manual):** synthetic 100-row JSONL, 50 optimizer steps, `save_steps=25` → loss decreases, no OOM on 5090, checkpoint saves and reloads, tokenizer/processor are saved alongside the adapter.
- **Inference smoke (manual):** load saved adapter via `FastVisionModel.from_pretrained(<adapter_dir>)` (or the equivalent base-model + `PeftModel.from_pretrained` two-step if Unsloth's loader doesn't auto-detect adapter dirs); run on `M01_001_001.tif` → produces a parseable `submit_estimate` call. Accuracy not asserted here; the only thing tested is wire compatibility.
- **RLVR handoff:** run `rlvr/train_grpo.py` smoke with `--sft-model out/sft/run0` → loads without error, env unit tests pass, 10-rollout no-optimizer smoke completes. **If the load path fails** (likely if `train_grpo.py:121` calls `FastVisionModel.from_pretrained` directly on an adapter directory without base-model context), update RLVR's loader to do the explicit base + adapter load, or have SFT also save a merged-but-quantized variant alongside the adapter. The handoff convention should be pinned in code on the SFT side and verified by this smoke run.
- **Pre-RLVR gate:** run the agent-loop eval (§10.2) on the saved adapter → confirm `tool_call_parseability_rate >= 0.80` before considering SFT done.
- **Existing harness regression:** `python -m pytest tests/` clean.
- **Lint / type:** `python -m ruff check models/langslice-gemma-4/training/sft/` clean; `python -m basedpyright models/langslice-gemma-4/training/sft/` clean.

## 12. Risks and follow-ups

- **Unsloth VLM SFT API surface.** Library moves fast; verify exact method names + argument shapes via Context7 before writing code (per memory note `feedback_verify_third_party_docs`). The implementation plan opens with this verification step.
- **Chat-template assistant-mask reliability for VLM tool traces.** `processor.apply_chat_template(..., return_assistant_tokens_mask=True)` is the load-bearing mechanism in §7.2. If Gemma 4's chat template does not implement the `{% generation %}{% endgeneration %}` markers correctly around tool-call output, the mask will be wrong and the manual-span fallback in collate.py kicks in. The implementation plan's first step verifies the template *and* the mask round-trip on a canned trace before any optimizer step.
- **LoRA adapter handoff to RLVR.** RLVR's `train_grpo.py:121` calls `FastVisionModel.from_pretrained(args.sft_model, ...)`. Whether this method auto-detects an adapter directory and loads it on top of the base model — versus requiring a `PeftModel.from_pretrained` second step — needs verification. If it doesn't auto-detect, the SFT save step writes both an adapter directory and a small `adapter_config.json` that points at the base model, and RLVR's loader is updated to do the explicit two-step load. Verified by the §11 RLVR-handoff smoke run.
- **Subject-level holdout integrity.** If `subject_id` is missing or malformed in the data sessions' JSONL, holdout becomes useless. `dataset.py` validation must reject any row with empty `subject_id`.
- **Atlas-version drift.** Different atlas versions (e.g. Allen CCFv2 vs CCFv3) have different mm origins. Data sessions stamp `atlas_version` per row; trainer trusts the data sessions and does not currently enforce same-version-only.
- **Sequence-length budget.** `max_seq_length = 16384` is the default; multi-turn multi-image traces may still overflow if the corpus contains long trajectories. Truncating mid-trace silently drops images or assistant turns and produces meaningless training signal. The collator must reject (with a clear error) any rendered example exceeding `max_seq_length` rather than silently truncating.
- **Training-time agent-loop eval cost.** ~5–10 min per pass at default cadence × baseline + every 200 steps. Can dominate wall-clock for short runs. Mitigation: raise `agent_eval_steps` to 400+ or downsample to 3 of M0[1-9].
- **Single-bucket coverage gap and parseability risk.** Without anatomy-grounding buckets *or* programmatic-skeleton coverage, the SFT corpus's only signal for tool format is whatever Gemini-distilled traces emit. If the parseability gate (§10.3) fails, the cheapest remediation is the deferred programmatic-skeletons bucket from the original spec — format-correct by construction, no LLM required to generate. **Keep that bucket spec-ready as a fallback even though it's cut from v1.**

## 13. Reuse pointers

- `src/langslice_harness/harness/estimation/prompts.py` — single-slice + group system prompts, used verbatim.
- `src/langslice_harness/harness/estimation/tools.py` — tool-name and arg signatures the renderer must produce.
- `src/langslice_harness/atlas/` — slice helpers (used by agent-loop eval callback to render comparison images).
- `references/TestImages/M0[1-9]/ground_truth.json` — ground-truth labels for the agent-loop eval callback.
- `requirements-rlvr.txt` — same library pins; no new requirements file.

## 14. Cleanup

Delete the following stubs flagged in `2026-05-04-gemma4-rlvr-training-design.md` §13 (now made redundant by this work):

- `models/langslice-gemma-4/training/finetune.py`
- `models/langslice-gemma-4/data/build_triplets.py`
- `models/langslice-gemma-4/data/distill_cot.py`
- `models/langslice-gemma-4/data/generate_atlas_slices.py`

## 15. Dependencies

| Dependency | Owner | Blocks | Notes |
|---|---|---|---|
| SFT JSONL corpus (single bucket — agent traces) | Data sessions | Real training | Renderer + dataset code can be built and unit-tested without it (canned fixtures suffice). |
| Gemma 4 E4B chat template + instruct-masking verification | Implementation | Real training | Must confirm tokenizer applies `enable_thinking=false` correctly and that loss masking aligns to assistant-turn token spans in the multimodal case. |
| `prompts.py` + `tools.py` | Already in tree | Real training | Production prompts; renderer imports rather than duplicates. |
| RLVR scaffold | Already in tree | Handoff verification | Integrates via `--sft-model`. May require a small change to `train_grpo.py`'s loader to do explicit base-model + adapter load if `FastVisionModel.from_pretrained` doesn't auto-detect adapter directories. Verified by `tests/test_sft_handoff.py`. |
| `references/TestImages/M0[1-9]` | Already in tree | Agent-loop eval callback | M01–M09 already labeled. |

## 16. Status

Design v1.3.

**Diff vs v1.2** (this revision): Data contract reverted from HF chat-template format back to a langslice-native trace format (§6) that mirrors the production agent loop's natural `tool_call → tool_result → ... → submit` shape. The HF translation moves into the renderer (§7.1) where it belongs. Data sessions emit a clean, readable trace; chat-template ceremony stays inside the trainer. Codex's structural concerns from v1.2 (`tool_call_id` pairing, content-block structure, `tools` schema) are still addressed — they're now generated by the renderer rather than carried in the JSONL.

**Diff vs v1.1** (Codex adversarial review fixes, retained): Renderer + collator split into two stages so the loss mask is computed *after* the processor expands image tokens. Dropped the `peft_config`-passed-to-`SFTTrainer` step (TRL hard error when the model is already a `PeftModel`). `max_seq_length` raised from 4K → 16K. Added pre-training base-model baseline eval (§10.1) and a hard pre-RLVR parseability gate at 0.80 (§10.3). Flagged LoRA-adapter-handoff verification (§12).

Implementation plan to be written next via the `writing-plans` skill. Hackathon deadline 2026-05-18.
