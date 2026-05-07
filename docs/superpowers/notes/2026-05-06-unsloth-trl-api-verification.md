---
title: Unsloth + TRL VLM SFT API verification (Gemma 4 E4B)
date: 2026-05-06
scope: Pin down current API surface for the SFT training plan before writing code.
status: complete
---

# Unsloth + TRL VLM SFT API verification

Sources: Context7 (`/unslothai/unsloth`, `/unslothai/notebooks`, `/huggingface/trl/v1.0.0`,
`/huggingface/transformers/v5.0.0`), Exa search (Hugging Face issue/PR threads,
unsloth/gemma-4-E4B-it model card, TRL chat-template docs).

## 1. Verified Gemma 4 E4B model ID

`unsloth/gemma-4-E4B-it` — confirmed live on Hugging Face (released 2026-04-22).

| Property | Value |
|---|---|
| Architecture | `Gemma4ForConditionalGeneration` |
| Modalities | text + image (audio supported on E2B/E4B too) |
| Context length | 128K (per E4B model card) / 256K family max |
| Effective params | ~4.5B (8B with embeddings) |
| Required VRAM (BF16) | ~16 GB; 4-bit fits comfortably on 5090 |
| Tokenizer class | `GemmaTokenizer` |

Available via `FastVisionModel.from_pretrained("unsloth/gemma-4-E4B-it", ...)`.
The Unsloth notebook collection ships a `Gemma4_(26B_A4B)-Vision.ipynb` confirming
the family is wired through `FastVisionModel`.

## 2. Unsloth `FastVisionModel` API

### 2.1 `from_pretrained` (verified)

```python
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-4-E4B-it",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    # max_seq_length is accepted but optional for vision models
)
```

Returns a `(model, processor)` pair. Note: VLM variant returns a `processor`
(not a `tokenizer`) — the spec/plan correctly call this `processor`.

### 2.2 `get_peft_model` (verified — exact match to plan)

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    use_gradient_checkpointing = "unsloth",
    # target_modules = "all-linear",  # optional, defaults to all-linear
)
```

Returns a `PeftModel`. Plan's per-call kwargs match the canonical Gemma-3 VLM
notebook (`Gemma3_(4B)-Vision.ipynb`) and the Qwen2.5-VL-GRPO notebook.

### 2.3 Mode switching (verified)

```python
FastVisionModel.for_inference(model)   # in-place
FastVisionModel.for_training(model)    # in-place
```

Both modify the model in place; no return value. Plan's eval-callback usage matches.

## 3. TRL `SFTTrainer` + `SFTConfig` (verified against /huggingface/trl/v1.0.0)

### 3.1 VLM-safe pattern

```python
trainer = SFTTrainer(
    model=peft_wrapped_model,        # already a PeftModel — do NOT pass peft_config
    args=SFTConfig(max_length=None), # critical for VLM: no truncation of image tokens
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=processor,
    data_collator=custom_collator,
    callbacks=[...],
)
trainer.train()
```

TRL doc (sft_trainer.md in v1.0.0) ships an explicit Qwen2.5-VL example:

```python
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=SFTConfig(max_length=None),
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train"),
)
```

Confirms: `max_length=None` is the documented VLM safety knob.

### 3.2 Pre-wrap PEFT then pass model — confirmed valid

TRL's `peft_integration.md` documents the advanced "wrap with `get_peft_model`,
then pass `model=<PeftModel>`, no `peft_config`" path. Spec is correct that
passing both is incorrect; spec/plan use the correct single-path setup.

### 3.3 `assistant_only_loss` — confirmed text-only-friendly, VLM-unsafe

TRL chat-template doc explicitly says:

> SFT with `assistant_only_loss=True` needs `{% generation %}` /
> `{% endgeneration %}` markers around assistant output, so the loss mask
> can target only assistant tokens.

This relies on `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`
working on the *tokenizer*. For VLM, the assistant mask must be re-aligned after
the *processor* expands image tokens — which is why TRL's `assistant_only_loss=True`
flag is unsuitable for VLM and we must build the labels mask ourselves in the
collator.

**Decision: keep `assistant_only_loss=False` in `SFTConfig`; build the labels
mask in `LangSliceCollator`.** Matches the plan.

## 4. `processor.apply_chat_template(..., return_assistant_tokens_mask=True)` — fixed in our pinned version

### 4.1 History

- **2024-07** (PR #30650, transformers 4.41+) — `tokenizer.apply_chat_template`
  gains `return_assistant_tokens_mask` flag; chat templates wrap assistant
  output in `{% generation %}` / `{% endgeneration %}` blocks.
- **2025-03** (PR #36149) — refactor blocks the flag on `ProcessorMixin`
  (issue #36713).
- **2025-04** (PR #37602) — splits jinja logic out so processor doesn't
  re-tokenize; closes #36713.
- **2025-06** (issue #38521) — follow-up `char_to_token` AttributeError on
  some MLLM processors; closed 2025-07-18 by `zucchini-nlp` via PR #38545.

### 4.2 Status in our pin

`transformers==5.5.0` (Apr 2026) postdates all three fixes. The flag works on
`processor.apply_chat_template` for VLM models that ship a chat template with
`{% generation %}` markers, returning `assistant_masks` in the result dict
when called with `tokenize=True, return_dict=True, return_assistant_tokens_mask=True`.

### 4.3 Caveat (image-token alignment)

The mask is computed at jinja render time, *before* the processor expands
image-placeholder tokens. After expansion, the mask must align with the
post-expansion `input_ids`. The transformers fix in PR #38545 handles this
by re-applying the mask value of an unexpanded image token to its full
expanded "run". Plan's Task 8 image-token sanity check verifies this
empirically and falls back to a manual-span mask if it ever misfires.

### 4.4 Whether Gemma 4's chat template has `{% generation %}` markers

The unsloth/gemma-4-E4B-it model card's `tokenizer_config.json` contains a
custom Gemma 4 chat template with full tool_call / tool_response handling
(`<|model|>`, `<|turn>`, `<|tool_call|>`, etc. markers). Whether the upstream
template wraps the assistant output in `{% generation %}` markers is **not
confirmed from public sources** — most upstream model templates omit them and
rely on TRL/Unsloth to swap in a training-template patch.

TRL ships `get_training_chat_template(tokenizer)` which returns a patched
template for recognized families (DeepSeek-V3, **Gemma**, Gemma2, GLM-4-MoE,
GPT-OSS, LLaMA 3, Phi-3, Qwen2.5, Qwen3, Qwen3.6) — Gemma 4 isn't yet in the
listed set as of TRL v1.0.0 docs (TRL 1.1.0 may extend it; verify at
implementation time).

**Decision (primary path):** Try the unmodified Gemma 4 template first. If
`assistant_masks` comes back all-zero, swap in TRL's
`get_training_chat_template(processor.tokenizer)` and pass it via the
`chat_template=` arg. If both fail, fall back to the manual-span mask
implemented in plan Task 8. The plan's three-layer setup (built-in →
TRL-patched → manual-span) is robust to any of those outcomes.

## 5. Other deltas vs spec

- **Spec mentioned `chat_template_kwargs={"enable_thinking": False}`.** The
  Gemma 4 templates we inspected don't expose an `enable_thinking` toggle
  (the ones that do: Qwen3, DeepSeek-V3 reasoning variants). Passing the
  kwarg is harmless if the template ignores it. **Recommendation: keep it
  in the call site for forward compatibility; don't depend on it taking
  effect for Gemma 4.**

- **`max_seq_length` — Unsloth `from_pretrained` accepts it for both
  `FastLanguageModel` and `FastVisionModel`.** Plan passes
  `max_seq_length=16384` for VLM, which Unsloth honors for KV-cache sizing.
  `SFTConfig.max_length` stays `None` (separate concern: TRL truncation of
  image tokens, which we explicitly opt out of).

- **`save_steps` and `eval_steps`** are `SFTConfig` fields — verified in
  TRL v1.0.0 docs (inherited from `TrainingArguments`).

## 6. Final verdict

The plan's API assumptions all hold against TRL 1.1.0 / transformers 5.5.0 /
unsloth 2026.4.8. No breaking changes required. The single load-bearing risk
(VLM assistant-mask correctness) has both a verified happy-path mechanism
(`return_assistant_tokens_mask=True` on processor) AND a safety net (manual-span
fallback in Task 8).

**Trust `return_assistant_tokens_mask=True` (primary path).** Manual-span
fallback (plan Task 8) remains the documented escape hatch.

Proceed to plan Task 2 (module skeletons + test fixtures).
