# langslice-gemma-4

Fine-tuned **Gemma 4 E4B** for brain-section position estimation, deployed as a drop-in replacement for Gemini inside the LangSlice estimation agent loop (`fetch_atlas` + `submit_estimate`).

## Authoritative design

The SFT data design is specified in `docs/superpowers/specs/2026-04-25-gemma4-sft-data-design.md`. Read it before changing anything in this directory. Earlier descriptions of "comparison-triplet SFT" with the 31B model are superseded by that spec.

## Approach (summary)

- **Base:** Gemma 4 E4B (efficient ~4B variant, multimodal, native function-calling per Unsloth).
- **Training shape:** narrow-task tool-use SFT — the model learns to operate the existing LangSlice position-estimation loop with constrained tools, broad-to-narrow atlas comparisons, and calibrated final submissions. Do not train mandatory bracket/verify/fine-tune narration as the core behavior.
- **Data scale:** 2K–15K SFT examples across five buckets (agent traces, landmark listing, bbox grounding, multi-slice morphology, programmatic skeletons). Spec §5 defines the buckets.
- **Thinking/rationale policy:** train deployment traces with Gemma thinking off by default. Keep Gemini rationale summaries as metadata and fallback/auxiliary caption data; do not make full reasoning traces the default target for E4B.
- **Augmentation pipeline:** atlas-image transforms (DAPI/Nissl/brightfield mimicry, resolution-shift, jitter) reused for SFT and later RLVR. Spec §9. **This is Phase 1 / the active blocker** as of 2026-04-25.
- **Validation:** SliceBench (in development) is the gate.
- **RLVR:** separately scoped, larger phase, brainstormed later.

## Hardware

- Fine-tuning: RTX 5090 (32 GB VRAM) — E4B QLoRA fits comfortably.
- Inference: same constraint envelope; quantized E4B runs locally on consumer GPUs.

## Directory layout

The current files in `data/`, `training/`, `inference/` are scaffolding from the earlier triplet-based plan and are being revised per the SFT spec. Do not treat their current contents as the implementation target.

- `data/` — slice extraction, augmentation pipeline, bucket-specific generators (per spec §5)
- `training/` — Unsloth QLoRA fine-tuning configs and runner
- `inference/` — local agent-loop runner using the fine-tuned model

## Status

Pre-implementation. Spec finalized 2026-04-25. Implementation plan pending — write-plans phase begins after the augmentation pipeline is built.
