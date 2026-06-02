# langslice-gemma-4-e4b

Public model card for the LangSlice Gemma 4 E4B variant.

## Hugging Face

- Canonical model repo: https://huggingface.co/greenpolo/langslice-gemma-4-E4B

This repository does not ship model weights or private datasets.

## Intended use

- Histology slice AP-coordinate estimation inside the LangSlice tool loop.
- Best used through the LangSlice harness CLI (`langslice estimate`, `langslice register`).

## Eval snapshot (v1 lane)

- Reference eval lane: single-slice AP estimation (SliceBench tiny/small workflows).
- Historic baseline summary captured in repo docs:
  - Gemini 3 Flash reference: mean MAE around 1.28 mm.
  - `docker-sft-1011-merged-bf16` SFT checkpoint: best-of-4 MAE around 1.15 mm, mean MAE around 2.88 mm.

Numbers above are copied from tracked training docs and should be refreshed when new public evals are published.

## Compatibility notes

- Public model-scoped launch commands:
  - `langslice-gemma-rl`
  - `langslice-gemma-sft`
- iSFT is retired as a public product/pipeline and has no public launcher.
- Transition shims prefer shared packages when available under:
  - `models/langslice-traces/langslice_traces`
  - `models/training-core/langslice_training`
  - `models/data/langslice_data`

## Safety and data policy

- No private histology corpora or raw manifest rows are committed here.
- No model checkpoints are committed here.
- Local-only caches and generated corpora are gitignored.
