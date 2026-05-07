# langslice-gemma-4

Fine-tuned **Gemma 4 E4B** for brain-section position estimation, deployed as a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate`).

## Authoritative Designs

- SFT data: `docs/superpowers/specs/2026-04-25-gemma4-sft-data-design.md`
- RLVR training: `docs/superpowers/specs/2026-05-04-gemma4-rlvr-training-design.md`

## Approach

- **Base:** Gemma 4 E4B, multimodal, trained for the existing LangSlice tool loop.
- **SFT:** narrow-task tool-use traces, landmark listing, bbox grounding, multi-slice morphology, and programmatic skeletons.
- **RLVR:** multi-turn GRPO in `training/rlvr/`, with one gated closeness reward on submitted coordinate accuracy.
- **Holdout:** RLVR uses deterministic subject-level train/eval splitting; no subject appears in both sets.

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
