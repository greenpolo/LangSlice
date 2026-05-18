# Expert Iteration SFT Driver

This directory implements the active training pipeline. Read this if you are
adding to, debugging, or running expert iteration.

For top-level orientation, see
[`models/langslice-gemma-4/training/README.md`](../README.md).

## What this is

**Expert iteration** = rejection-sampling fine-tuning. Loop:

1. Take the current best Gemma 4 LoRA adapter.
2. Run inference (rollouts) on N prompts × M generations each, via vLLM.
3. Score every (prompt, gen) pair against ground truth.
4. Filter: keep best-of-N per prompt (or threshold-accept rollouts ≤ tolerance).
5. Convert ADK event traces to langslice-native SFT trace JSONL.
6. Append kept slices to the iterative corpus.
7. Build a unioned corpus = (sample of distilled corpus) ∪ (cumulative iterative).
8. SFT-retrain from the original SFT adapter on the unioned corpus.
9. Slicebench eval. Compute curriculum weights for the next round.
10. Repeat.

This is the pivot from GRPO RLVR (which regressed our model — see
`../rlvr/README.md`).

## Files

| File | Purpose |
|---|---|
| `iterate.py` | Main multi-round driver. CLI entrypoint. |
| `rollout.py` | `ThreadPoolExecutor` fan-out around `run_single_slice_session`. |
| `filter.py` | Pure best-of-N + threshold-accept logic. |
| `trace_format.py` | ADK event log → langslice-native SFT trace dict (reuses `_convert_trace`/`_classify_quality`/`_trim_trace` from `_local/trace_collection/build_sft_corpus.py`). Emits top-level `section_id` (Phase 6). |
| `state.py` | `RunState` dataclass; phase-level resumability across rounds. |
| `vllm_lifecycle.py` | docker-compose vLLM start/stop, LoRA merge helper, REST adapter hot-swap helpers. |
| `path_rewriter.py` | Stage corpus images into a unified tree under `<output_dir>/{queries,atlas}/`; rewrite JSONL row paths; supports `base_sample_n` for option C. |
| `synth_corpus.py` | **Phase 3.** Synthetic terminal-trace generator: takes section specs, generates `CanonicalTrace` via `langslice_traces.generate_trace(strategy="lane_a_prefix")`, injects a synthetic `FinalAnswer(position_mm=GT)`, renders as answer-only or full multi-turn. Writes JSONL rows tagged `quality.source="synthetic_terminal"`. |
| `seen_ledger.py` | **Phase 4.** Section-level "seen" ledger with SHA-256 manifest fingerprint. Refuses to load on shard/override drift (raises `StaleLedgerError`). |
| `configs/round_default.toml` | Documented defaults (informational; iterate.py is CLI-driven). |

Tests in `tests/test_expert_iteration_*.py`. **All 81 tests pass** (run via
`docker compose ... run --rm training python -m pytest tests/test_expert_iteration_*.py`).

## How to run

> **Host-side invocation requires `PYTHONPATH=models/langslice-gemma-4/training`**
> (PowerShell: `$env:PYTHONPATH = "models/langslice-gemma-4/training"`;
> Bash: `PYTHONPATH=models/langslice-gemma-4/training python -m iSFT.iterate ...`).
> The package is not at the repo root — `pyproject.toml` only wires this path
> for `pytest` collection. The single_turn_rl README documents the same convention.
> When iterate.py shells out to `train_sft.py` inside the Docker container, the
> container's own PYTHONPATH covers the subprocess.

### Recommended invocation (combines all the wins)

```powershell
$env:PYTHONPATH = "models/langslice-gemma-4/training"
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = "out/expert_iteration/run-$ts"
python -m iSFT.iterate `
  --base-checkpoint out/sft/docker-sft-1011-merged-bf16 `
  --base-corpus models/langslice-gemma-4/data/sft_examples.jsonl `
  --iterative-corpus-dir out/iterative_sft `
  --allocation-root data/manifest `
  --output-dir $runDir `
  --rounds 1 --rollouts-per-prompt 4 --prompts-per-round 30 `
  --filter-mode best-of-n --apply-clahe `
  --manage-vllm --vllm-lora-mode `
  --vllm-base-compose docker-compose.training.yml `
  --vllm-url "http://127.0.0.1:8000/v1" --vllm-max-model-len 16384 `
  --concurrency 4 --temperature 0.9 `
  --distilled-sample-n 150 --distilled-sample-seed 42 `
  --sft-initial-adapter out/sft/docker-sft-1011-trimmed-noeval `
  --eval-bench small --eval-num-generations 1 `
  --run-id "run-$ts"
```

Iterate.py runs **on the host** (not inside Docker). It dispatches docker
compose subprocess calls for vLLM lifecycle and SFT retrain, and runs rollout
HTTP calls directly via litellm.

### Critical flags

| Flag | What it does | When to use |
|---|---|---|
| `--rollouts-per-prompt N` | How many generations per prompt (best-of-N pool) | 4 is the sweet spot per slicebench measurements |
| `--prompts-per-round N` | Total prompts to roll out per round | 30 for smoke, 800-2000 for real runs |
| `--filter-mode best-of-n` | Keep argmin(error) per prompt | Default, lowest-variance signal |
| `--filter-mode threshold-accept` | Keep all rollouts ≤ `--threshold-pct` plane error | Stricter; produces fewer but higher-quality traces |
| `--filter-mode both` | Union of best-of-N and threshold-accept | Most data per round |
| `--apply-clahe` | Use SFT-corpus-matched 25% CLAHE preprocessing | **Always on** to match SFT pixel distribution |
| `--manage-vllm` | Spawn/teardown vLLM via docker compose | **Required for multi-round** |
| `--vllm-lora-mode` | Skip per-round merge step; hot-swap LoRA via REST | **Saves 3-5 min per round.** Refuses to combine with `--sft-initial-adapter` (double-LoRA guard). |
| `--vllm-base-model-path /models/sft-base` | Make vLLM serve from the `langslice-models-fast` volume instead of the bind-mounted `--base-checkpoint` | **Saves ~7 min per vLLM cold start** (10 min → 2-3 min). Only set when the fast volume already holds an identical copy of the base. |
| `--vllm-max-model-len 16384` | vLLM context window | Default 8192 is too tight; some tool-call rollouts hit 12-13K |
| `--distilled-sample-n N` | Sample N rows from distilled corpus per round (option C) | Use 100-300; saves SFT retrain time vs all 1011 |
| `--sft-initial-adapter PATH` | **Resume SFT from existing adapter** (option D) | **Mandatory for small per-round corpora** — fresh-init destroys the model. Mutually exclusive with `--vllm-lora-mode` (use `sft_iterative.toml` instead). |
| `--eval-bench small` | Slicebench bench size | "small" (n=317) for real runs, "tiny" (n=64) for smoke |
| `--eval-num-generations 1` | Single-shot eval per slice | Use 4 for variance info, 1 for speed |
| `--rounds N` | Multi-round chaining | Each round is ~1.5 hr (smoke) to ~5 hr (real) |
| `--start-round K` | Resume from round K | State.json detects last completed phase |

### Speed/diversity flags (Phase 2–5)

| Flag | What it does | When to use |
|---|---|---|
| `--atlas-embedding-cache DIR` | Splice cached SigLIP atlas embeddings into the forward pass; threaded through to `train_sft.py`. | Always, once the cache is precomputed. Bit-exact-verified. |
| `--query-embedding-cache DIR` | Sibling of `--atlas-embedding-cache` for per-row slice images. Threaded through to `train_sft.py`. | Always, once precomputed. Combines with atlas cache for full SigLIP-skip coverage. |
| `--synthetic-per-round N` | Generate N synthetic terminal-trace rows per round via `synth_corpus.py` and append to the round's iterative corpus. Default 0 (no synthesis). | When the real-rollout corpus alone is too small. Synthetic rows use real manifest query images + injected GT submit. |
| `--multi-turn-floor F` | Fraction of synthetic rows that render as full multi-turn (`sft_full`); the rest render as answer-only. Default 0.10. | Lower means more direct visual→AP training. Raise if tool-calling regresses at inference. |
| `--synthetic-reasoning-mode MODE` | `region_dump` preserves legacy submit `reasoning`; `thinking_signature` emits Gemma thinking rows with compact atlas geometry clues and turn-splits multi-turn traces. Default `region_dump`. | Use `thinking_signature` for the native-thinking ablation. Keep `region_dump` for byte-compatible baseline runs. |
| `--bucketed-shape-sampler` | Forward to `train_sft.py`. Yields shape-pure micro-batches (answer-only vs multi-turn never mixed). | **Required when `--synthetic-per-round > 0`** with any mix — otherwise per-step loss variance spikes. |
| `--use-seen-ledger` | Activate section-level seen ledger; sections selected in any prior round are skipped. | When making >1 pass over the corpus across many rounds. |
| `--ledger-path PATH` | Override ledger file location. Default: `<output_dir>/seen_ledger.jsonl`. | Cross-run sharing. |
| `--reset-ledger` | Delete the ledger file before this run. | When restarting fresh. |
| `--adaptive-accept` | Replace the static `--threshold-pct` accept rule with a quantile-derived threshold from `adaptive.schedule.AdaptiveSchedule`. | When MAE drifts substantially round-over-round. |
| `--adaptive-accept-quantile Q` | Quantile of recent errors to set the threshold at. Default 0.95. | Lower = stricter accept (fewer rows kept, higher quality). |
| `--adaptive-accept-warmup-n N` | Use the static threshold until N observations have been gathered. Default 50. | Avoids degenerate threshold from a tiny warmup window. |
| `--curriculum-weights-dir DIR` | Per-round `round_<k>_weights.json` files bias the v2 sampler toward weak coord bins. | Multi-round runs once Phase 4 ledger + slicebench `per_coord_bin` history is available. |

### Per-round phases

Each round runs through these phases. State.json tracks which is done so
crashes can resume from the next phase:

1. `sampled` — pick N prompts from RLVR allocation (curriculum-weighted if active; uses the **v2 bin-aware sampler** when `--use-seen-ledger` is on, marks all picked sections as seen at sample time)
2. `rollouts` — fan out N×M rollouts via vLLM
3. `scored` — apply position reward to each (prompt, gen) pair
4. `filtered` — apply best-of-N or threshold-accept (or adaptive-quantile threshold when `--adaptive-accept` is on)
5. `appended` — write kept slices to `out/iterative_sft/round_<k>.jsonl`, then (if `--synthetic-per-round > 0`) generate + append synthetic terminal-trace rows. Crash-resume detects partial synthetic writes via `quality.source="synthetic_terminal"` count and skips re-generation.
6. `unioned` — build unified corpus + stage images + validate
7. `trained` — SFT retrain via train_sft.py subprocess (`--bucketed-shape-sampler` recommended when synthetic rows are in the mix)
8. `evaluated` — slicebench small/tiny eval
9. `curriculum` — recompute per-bin weights for next round (if curriculum dir set)

## What's working vs known issues

**Working:**
- Full pipeline runs end-to-end (verified smoke v3, 2026-05-09)
- vLLM lifecycle including LoRA hot-swap mode
- Path rewriting + image staging across rounds
- Resumability via state.json
- Curriculum integration is wired but uses uniform weights until a
  `--curriculum-weights-dir` is passed

**Known issues / things to know:**
- Rollouts on the 27k allocation pool sometimes hit context limits at 12K
  even with `--max-iterations` cap. Recommendation: run with
  `--vllm-max-model-len 16384` to give more headroom.
- vLLM Gemma 4 + multimodal LoRA works in our 0.16.1.dev0 build
  (Unsloth cherry-picked from 0.19.0). REST endpoints are
  `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`.
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var is required.
- vLLM cold-start is ~2-3 min from the named volume `langslice-models-fast`,
  vs ~10 min from the bind mount. The Docker volume is configured in
  `docker-compose.training.yml`; populate it once with
  `docker volume create langslice-models-fast` + a one-shot copy from
  `out/sft/docker-sft-1011-merged-bf16/` to `/sft-base/` inside the volume.
- The SFT phase requires the GPU, so vLLM has to come down between rollout
  and SFT phases. After SFT, when vLLM comes back up, the same base file is
  page-cached → second cold-start is faster (~1 min).

## Diagnosing a failed run

Look at, in order:
1. `<output_dir>/state.json` — which phase failed?
2. `<output_dir>/round_<k>_rollouts.json` — how many rollouts succeeded?
3. `out/iterative_sft/round_<k>.jsonl` — were any slices kept after filtering?
4. `<output_dir>/round_<k>_corpus.jsonl` — did the corpus union succeed?
   Look at the `union_stats` block in state.json for `rows_dropped_missing_images` —
   that field nonzero = path resolution issue (pre-2026-05-09: Windows colon bug).
5. `<output_dir>/round_<k>_slicebench/summary.json` — eval result.

For training catastrophes (model destroyed, MAE > 5mm), check that you
passed `--sft-initial-adapter` — without it, fresh-init on a small corpus
will trash the model.

## Why option D matters

The original plan used "fresh LoRA each round on (full distilled + iterative)
corpus." That works in principle but takes ~1 hr per round on the full
distilled corpus.

We tried "fresh LoRA each round on (sampled distilled + iterative)" (option C)
to speed up — and it catastrophically failed. The fresh LoRA on 180 slices
for 63 steps is essentially still random when training stops, and a random
LoRA on a merged base = corrupted forward pass = MAE 15mm.

Option D (`--sft-initial-adapter`) loads the existing SFT adapter and
**continues** training it. The adapter starts converged (from the original
1011-example SFT) and gets nudged toward the iterative corpus. Mathematically
correct, ~10× faster than option A, and stable.

## Future work (left for the next agent)

- **Atlas + query embedding splice live (Phase 2, 2026-05-11)** — both
  caches are wired. Need a one-time precompute on GPU before activating;
  see [`../embeddings/README.md`](../embeddings/README.md).
- **Curriculum (Phase 6 lift)** — per-coordinate-bin weight updates wired
  via `--curriculum-weights-dir`. Section_id-keyed canonical. See
  [`../adaptive/curriculum/README.md`](../adaptive/curriculum/README.md).
- **Synth data ingestion** — when the synthetic data pipeline lands, drop
  rows into `data/manifest/shards/<plane>/synthetic.jsonl` and the bin
  computation auto-picks them up. The Phase 3 `synth_corpus.py` generates
  procedural traces over real manifest images and is the throughput path.
- **vLLM-keep-warm across SFT phase** — currently we tear down vLLM for SFT
  due to GPU contention. If we ever go multi-GPU we could keep both running.

## Phase-by-phase change log (iSFT speed upgrade, 2026-05-11)

| Commit | Phase | What landed |
|---|---|---|
| `b8e8508` | 1 | Repo restructure: `tools/expert_iteration/` → `iSFT/`; embeddings tools folded in; `langslice_traces/`+`langslice_rlvr/` moved under `training/`. |
| `e9f8ac2` | 2 | `QueryEmbeddingCache` wired into `sft/collate.py`; `--query-embedding-cache` on both `iterate.py` and `train_sft.py`. |
| `3a26430` | 3 | `synth_corpus.py`; `--synthetic-per-round`, `--multi-turn-floor`. |
| `cc74892` | 4 | `seen_ledger.py`; `_sample_specs_v2` (bin-aware); `--use-seen-ledger`/`--ledger-path`/`--reset-ledger`. |
| `1b2c929` | 5 | `adaptive/schedule.py` lift; `--adaptive-accept`. |
| `67c7307` | 6 | `adaptive/curriculum/` lift; `section_id` keying through dataset + sampler. |
| `7a8b8ca` | post-review | Bucketed shape sampler + double-LoRA assertion + multi-turn-floor default 0.25→0.10. |
| `6973066` | post-review | Synthetic-row crash-resume guard + mark-seen at pick time. |
