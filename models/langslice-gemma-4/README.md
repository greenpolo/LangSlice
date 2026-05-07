# langslice-gemma-4

Fine-tuned **Gemma 4 E4B** for brain-section position estimation, deployed as a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate`).

## Authoritative Designs

- SFT training/input contract: `docs/superpowers/specs/2026-05-05-gemma4-sft-training-design.md`
- Historical SFT data design: `docs/superpowers/specs/2026-04-25-gemma4-sft-data-design.md`
- RLVR training: `docs/superpowers/specs/2026-05-04-gemma4-rlvr-training-design.md`

## Approach

- **Base:** Gemma 4 E4B, multimodal, trained for the existing LangSlice tool loop.
- **SFT:** v1 trains only on strict-accepted single-slice agent traces, supplied as a langslice-native JSONL corpus. See `training/sft/README.md`.
- **RLVR:** multi-turn GRPO in `training/rlvr/`, with one gated closeness reward on submitted coordinate accuracy.
- **Holdout:** RLVR uses deterministic subject-level train/eval splitting; no subject appears in both sets.

## SFT Corpus Handoff

The SFT trainer does not read per-run `raw_trace.json` files directly. Build one corpus JSONL first:

```bash
models/langslice-gemma-4/data/sft_examples.jsonl
```

Each row is a single-slice langslice-native trace with relative image paths. The corpus builder should walk verified/reroll trace directories, select the best strict-accepted run per id, copy or stage referenced images under the JSONL parent directory, then emit the JSONL. The trainer validates and renders that JSONL into Gemma chat-template messages at training time.

## RLVR Launch

From the repo root:

```bash
python -m langslice_rlvr \
  --config models/langslice-gemma-4/training/configs/grpo_pilot.toml \
  --sft-model out/sft/gemma4-e4b-langslice \
  --output-dir out/rlvr/phase_a \
  --test-images-root references/TestImages
```

Phase B resumes the Phase A LoRA adapter:

```bash
python -m langslice_rlvr \
  --config models/langslice-gemma-4/training/configs/grpo_phase_b.toml \
  --sft-model out/sft/gemma4-e4b-langslice \
  --resume-from-adapter out/rlvr/phase_a \
  --output-dir out/rlvr/phase_b \
  --test-images-root references/TestImages
```

## Directory Layout

- `data/` - slice extraction, augmentation pipeline, bucket-specific generators.
- `training/` - Unsloth QLoRA configs and runners.
- `training/rlvr/` - RLVR environment, dataset, reward, atlas grid, and GRPO driver.
- `inference/` - local agent-loop runner using the fine-tuned model.
