# Gemma 4 Training Pipeline — Agent Orientation

Read this first if you are a new agent working on this codebase. It tells you
what's here, what works, what's parked, and where to look next.

## Current state (2026-05-09)

We are training **Gemma 4 E4B** (8B vision-language model) for **brain slice
position estimation** — given a histology image of a brain section, the model
predicts where in the atlas (mm coord) it sits. Hackathon deadline is
2026-05-18.

**Trained models so far** (best → worst on slicebench tiny num_gens=4 best-of-4):

| Checkpoint | best-of-4 MAE | mean MAE | Failure rate | Status |
|---|---|---|---|---|
| Gemini 3 Flash (reference, API) | n/a | 1.28mm | n/a | external baseline |
| **`out/sft/docker-sft-1011-merged-bf16`** (current best) | **1.15mm** | 2.88mm | 3.5% | distilled SFT, 1011 examples |
| Pre-SFT base (`unsloth/gemma-4-E4B-it`) | 1.31mm | 2.57mm | 0.0% | shockingly competitive |
| `out/rlvr/stage1-n8-overnight` (300-step GRPO) | n/a | 15.26mm (small bench) | 55% | **regressed**, archived |

The SFT-merged-bf16 is the **anchor checkpoint** all expert-iteration rounds
should resume from. Don't overwrite it.

## Pipeline directory layout

```
models/langslice-gemma-4/training/
├── sft/              # Foundational SFT trainer (working). README in there.
├── rlvr/             # GRPO RLVR trainer (PARKED — see rlvr/README.md).
├── curriculum/       # Dynamic difficulty sampler (DORMANT, flag-gated off).
├── embeddings/       # Atlas SigLIP embedding cache (DORMANT, flag-gated off).
├── configs/          # TOML configs for SFT/GRPO runs.
└── unsloth_compiled_cache/   # Unsloth's compiled Triton kernels (do not edit).

tools/expert_iteration/   # Expert-iteration SFT driver (active pivot, see README).
slicebench/              # Eval tool (working). per-bin breakdowns + tok/s metrics.
```

Each subdirectory has its own README explaining its scope and current status.

## Training journey (the short version)

1. **SFT on 1011 distilled Gemini-3-Pro traces** → produced
   `docker-sft-1011-merged-bf16`. This is our base. Works.
2. **GRPO RLVR on the SFT base** → ran 300 steps, reward stayed flat
   (mean 0.24, KL grew 17×). Resulting checkpoint is **worse than the SFT
   base** on slicebench: 55% failure rate, MAE 15mm (vs 2.88mm). Pivoted away.
3. **Pivot: Expert-iteration SFT (rejection-sampling)** — generate N rollouts
   per prompt with current best model, keep the best ones, retrain on those,
   repeat. This is where active work is happening. See
   [`tools/expert_iteration/README.md`](../../../tools/expert_iteration/README.md).
4. First expert-iteration smoke (option C: fresh-LoRA on small corpus)
   regressed the model to MAE 15mm. Diagnosed: fresh LoRA + 180 slices ×
   63 steps = under-trained adapter. **Option D fix shipped**:
   `--initial-adapter` flag in `sft.train_sft` resumes training of an
   existing PEFT adapter rather than attaching a fresh one. Combine with
   `--sft-initial-adapter` in `iterate.py`.

## How to run the active pipeline (expert iteration)

```powershell
python -m tools.expert_iteration.iterate `
  --base-checkpoint out/sft/docker-sft-1011-merged-bf16 `
  --base-corpus models/langslice-gemma-4/data/sft_examples.jsonl `
  --iterative-corpus-dir out/iterative_sft `
  --allocation-root data/manifest `
  --output-dir out/expert_iteration/run_<timestamp> `
  --rounds 1 --rollouts-per-prompt 4 --prompts-per-round 30 `
  --filter-mode best-of-n --apply-clahe `
  --manage-vllm --vllm-lora-mode `
  --vllm-base-compose docker-compose.training.yml `
  --vllm-url "http://127.0.0.1:8000/v1" --vllm-max-model-len 16384 `
  --concurrency 4 --temperature 0.9 `
  --distilled-sample-n 150 --distilled-sample-seed 42 `
  --sft-initial-adapter out/sft/docker-sft-1011-trimmed-noeval `
  --eval-bench small --eval-num-generations 1
```

Full flag reference in `tools/expert_iteration/README.md`.

## Eval baseline (slicebench tiny num_gens=4 temp=0.9)

When iterating, compare against:
- **SFT base, all-samples MAE: 2.88mm** (we want to beat this)
- **SFT base, best-of-4 MAE: 1.15mm** (true ceiling — what RL/iteration aspires to)

Per-plane numbers from the SFT eval are in
`slicebench/runs/tiny/langslice-ft-numgen4/summary.json`.

## Hardware reality

- Single RTX 5090 (32 GB VRAM, Blackwell SM120)
- Docker Desktop on Windows 11 with WSL2 backend
- 128 GB host RAM
- WSL2 memory configured to 110 GB via `~/.wslconfig`
- vLLM cold start from bind-mounted model = slow (~3-10 min). A
  Docker named volume `langslice-models-fast` holds a copy of the SFT base
  for fast re-loads (~2-3 min).
- vLLM 0.16.1.dev0+g89a77b108 (Unsloth-cherry-picked Gemma 4 + LoRA support)
- Unsloth 2026.5.2, transformers 5.5.0, torch 2.10.0+cu128
- DO NOT upgrade torch to 2.11 — Unsloth maintainers explicitly say no
  ([issue #4851](https://github.com/unslothai/unsloth/issues/4851))

## Things that bit us (read this if you're about to repeat them)

1. **vLLM Gemma 4 GRPO is unsupported** — `fast_inference=False` is mandatory.
   Don't try to enable vLLM in GRPO. Background:
   <https://unsloth.ai/docs/models/gemma-4/train>.
2. **Multi-token prediction (MTP) is inference-only and text-only** — won't
   speed up our training or vision rollouts. Skip.
3. **Windows colon-in-filename bug** — section_ids with `:` get truncated by
   NTFS as Alternate Data Streams. Fixed in `_safe_id` in
   `_local/trace_collection/build_sft_corpus.py:86`. Always use that helper
   when staging files keyed by section_id.
4. **Fresh LoRA on small corpus = catastrophic** — never `get_peft_model()` a
   fresh LoRA and train it for <500 steps on <300 examples. The adapter is
   essentially still random when training stops; a random LoRA on a merged
   base produces garbage. Use `--initial-adapter` to resume an existing
   adapter instead.
5. **`LANGSLICE_LITELLM_PROXY_BASE` env var** — must point at the live vLLM
   for `litellm-proxy:<alias>` model strings to resolve. iterate.py auto-sets
   this from `--vllm-url` since 2026-05-09. If running other harness scripts
   manually, set it yourself.

## Where to find baselines

- `slicebench/runs/tiny/langslice-ft-numgen4/summary.json` — SFT n=64 num_gens=4
- `slicebench/runs/tiny/gemma-4-e4b-it-base/summary.json` — pre-SFT base, n=64 num_gens=4
- `slicebench/runs/tiny/gemini-3-flash-preview/summary.json` — Gemini reference

The `summary.json` schema is in `slicebench/score.py`. Look at the `overall`,
`per_plane`, `per_coord_bin`, and (if num_generations>1) `best_of_n` /
`mean_of_n` blocks. Decode tok/s metrics also live in `overall`.
