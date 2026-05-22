# langslice-gemma-4

Fine-tuned **Gemma 4 E4B** for brain-section position estimation, deployed as a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate` / `submit_group_estimate`).

## Training Docs

- Public training overview: `docs/training_overview.md`
- SFT data contract and trainer usage: `training/sft/README.md`
- Active single-turn RL trainer: `models/training-core/langslice_training/rl/single_turn/`

## Approach

- **Base:** Gemma 4 E4B, multimodal, trained for the existing LangSlice tool loop.
- **SFT:** v1 trains only on strict-accepted single-slice agent traces, supplied as a langslice-native JSONL corpus. See `training/sft/README.md`.
- **RL:** single-turn GRPO in `models/training-core/langslice_training/rl/single_turn/`, with one gated closeness reward on submitted coordinate accuracy.
- **Holdout:** RLVR uses deterministic subject-level train/eval splitting; no subject appears in both sets.

## SFT Corpus Handoff

The SFT trainer does not read per-run `raw_trace.json` files directly. Build one corpus JSONL first:

```bash
models/langslice-gemma-4/data/sft_examples.jsonl
```

Each row is a single-slice langslice-native trace with relative image paths. The corpus builder should walk verified/reroll trace directories, select the best strict-accepted run per id, copy or stage referenced images under the JSONL parent directory, then emit the JSONL. The trainer validates and renders that JSONL into Gemma chat-template messages at training time.

## Preferred SFT Launch

Use Docker for SFT training:

```powershell
models/langslice-gemma-4/training/scripts/docker/smoke.ps1
models/langslice-gemma-4/training/scripts/docker/sft.ps1 -RunName docker-run0
```

Native Windows SFT remains available through `models/langslice-gemma-4/training/_run_sft_msvc.cmd`, but Docker is preferred for CUDA/Unsloth performance and fewer Windows-specific dependency issues.

## iSFT status

iSFT is retired as a public LangSlice product/pipeline; there is no public iSFT launcher.

## Directory Layout

- `data/` - slice extraction, augmentation pipeline, bucket-specific generators.
- `training/` - Unsloth QLoRA configs and runners.
- `models/training-core/langslice_training/rl/single_turn/` - single-turn RL environment, dataset, reward, and eval tooling.
- `inference/` - local agent-loop runner using the fine-tuned model.
- `variants/langslice-gemma-4-e4b/README.md` - public model card + Hugging Face pointer (weights not stored in this repo).

## Compatibility and migration

- Preferred launch commands: `langslice-gemma-sft` and `langslice-gemma-rl`.
- During transition, training/data imports can resolve from shared package roots when present:
  `models/langslice-traces`, `models/synthdata`, `models/training-core`, `models/data`.
