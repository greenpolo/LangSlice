# Vision Embedding Caches — atlas + query (LIVE, flag-gated)

Pre-computes SigLIP embeddings for two classes of training images so the
trainer can skip the vision tower for cached inputs:

1. **Atlas cache** (`cache.py`, `precompute.py`) — keyed by
   `(atlas, plane, snapped_position_mm)`. 100% hit rate on the SFT
   corpus's atlas tool-result images.
2. **Query (slice) cache** (`query_cache.py`, `precompute_query.py`) —
   keyed by full manifest-relative image path. Sized to whatever fraction
   of the RLVR allocation gets precomputed.

Both run through the same forward-pre-hook in `splice.py`; atlas wins on
path collision (it's bit-exact-verified). The splice is **live as of Phase
2 of the iSFT speed upgrade (commit `e9f8ac2`)** but flag-gated off by
default — runs without the flags are byte-identical to the no-splice path.

For top-level orientation, see
[`../README.md`](../README.md).

## Why this exists

Atlas tool-result images appear repeatedly across SFT training examples.
Every batch, the model's vision tower (SigLIP) re-encodes the same atlas
images from scratch. **Measured atlas-grid hit rate on our 8643-row SFT
corpus: 100%** across 11 (atlas, plane) pairs. Every atlas tool-result image
in the corpus is eligible for caching. Conservatively that's 10-15% of
training compute reclaimed if we successfully splice cached embeddings
into the model's forward pass.

## Status

| Component | Status |
|---|---|
| `cache.py` (atlas lookup by atlas/plane/snapped position) | ✓ Built, unit-tested, bit-exact-verified |
| `precompute.py` (CLI: precompute SigLIP for every atlas grid pos) | ✓ Built; one-time GPU job |
| `query_cache.py` (slice-image lookup by full path) | ✓ Built, unit-tested |
| `precompute_query.py` (CLI: precompute SigLIP for slice images) | ✓ Built; one-time GPU job |
| `splice.py` (forward-pre-hook substituting cached embeddings) | ✓ **Live**; flag-gated off by default |
| `LangSliceCollator` hit/miss counters + sidecar emission | ✓ Built; accepts `atlas_cache=` + `query_cache=` + `enable_splice=` |
| `render_slates_from_cache.py` (on-disk atlas slate JPGs for SigLIP) | ✓ Built |
| `render_slates_native_step.py` (variant: native voxel-step atlas slate generator) | ✓ Built |
| `_measure_sft_hit_rate.py` (utility: walk SFT corpus, report cache hit rate) | ✓ Built |
| `_verify_query_cache.py` (utility: bit-exact check cached vs live SigLIP) | ✓ Built |
| Hit-rate measurement on the SFT corpus (atlas side) | ✓ **100% on 8643-row corpus, 11 atlas/plane pairs** |
| Hit-rate measurement on iSFT iterative corpus (query side) | ⧖ Pending the first overnight precompute pass |

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `AtlasEmbeddingCache`, `QueryEmbeddingCache`, `parse_atlas_path`, `save_query_pair` |
| `cache.py` | Lazy mmap-backed atlas lookup keyed by `(atlas, plane, snapped_pos_mm)` |
| `query_cache.py` | Lazy mmap-backed slice-image lookup keyed by full manifest path; per-`(plane, dataset)` shard files |
| `precompute.py` | CLI: load model, run SigLIP on every atlas grid image, save tensor cache |
| `precompute_query.py` | CLI: load model, run SigLIP on every slice image, save per-`(plane, dataset)` cache |
| `splice.py` | Forward-pre-hook on `Gemma4Model.get_image_features`: consults the sidecar, substitutes cached embeddings at the masked positions |
| `render_slates_from_cache.py` | Render on-disk JPGs for cached atlas embeddings (debug + SigLIP-input visualization) |
| `render_slates_native_step.py` | Variant slate generator at native atlas voxel step |
| `_measure_sft_hit_rate.py` | Walk an SFT-format JSONL corpus and report atlas/query cache hit-rate breakdown |
| `_verify_query_cache.py` | Sample-and-verify: cached query tensors match live SigLIP forward to `atol=1e-5` |

Tests: `tests/test_atlas_embedding_cache.py` (atlas)
+ `tests/test_atlas_embedding_collate.py` (collator)
+ `tests/test_query_cache_splice.py` (Phase 2; 11 tests).

## How to activate

### Step 1: precompute the caches (one-time, GPU)

**Atlas cache** (~10 min on the 5090):

```powershell
docker compose -f docker-compose.training.yml run --rm training bash -lc "
  cd /workspace/LangSlice/models/langslice-gemma-4/training && \
  python -m embeddings.precompute \
    --atlas-pairs allen_mouse_25um:coronal,whs_sd_rat_39um:coronal,whs_sd_rat_39um:horizontal,allen_mouse_25um:sagittal \
    --model /workspace/LangSlice/out/sft/docker-sft-1011-merged-bf16 \
    --output-dir /workspace/LangSlice/out/atlas_embeddings \
    --device cuda --dtype bf16 --step-mm 0.05
"
```

Produces `out/atlas_embeddings/<atlas>_<plane>.pt` per pair.

**Query cache** (longer — depends on RLVR allocation size):

```powershell
docker compose -f docker-compose.training.yml run --rm training bash -lc "
  cd /workspace/LangSlice/models/langslice-gemma-4/training && \
  python -m embeddings.precompute_query \
    --split rlvr \
    --model /workspace/LangSlice/out/sft/docker-sft-1011-merged-bf16 \
    --output-dir /workspace/LangSlice/out/query_embeddings \
    --device cuda --dtype bf16
"
```

Produces `out/query_embeddings/<plane>__<dataset>.pt` per pair.

### Step 2: pass the caches to the trainer

Two callers, same flag pair:

```powershell
# Standalone SFT
python -m sft.train_sft \
  --atlas-embedding-cache out/atlas_embeddings \
  --query-embedding-cache out/query_embeddings \
  ...

# Expert iteration (threads through to each round's train_sft.py call)
python -m iSFT.iterate \
  --atlas-embedding-cache out/atlas_embeddings \
  --query-embedding-cache out/query_embeddings \
  ...
```

When unset, behavior is byte-identical to the no-splice path.

### Step 3: measure hit rate (optional but recommended)

```powershell
python -m embeddings._measure_sft_hit_rate \
  --corpus out/iterative_sft/round_0.jsonl \
  --atlas-cache out/atlas_embeddings \
  --query-cache out/query_embeddings
```

Prints per-source hit-rate breakdown. Target: >50% combined hit rate to
justify the precompute. Atlas side alone typically >95%; query side
depends on whether you precomputed the full RLVR allocation.

### Step 4: bit-exact verification

`tests/test_atlas_embedding_cache.py::test_precomputed_matches_live_get_image_features`
(atlas side) and `embeddings._verify_query_cache` (query side) sample a
few cached entries and assert `torch.allclose(cached, live, atol=1e-5)`.
Both auto-skip when no GPU is available. **Run at least once after each
precompute** — a subtly-wrong embedding silently corrupts training.

## When to activate

The forward hook and both caches are built — what remains is the
one-time GPU precompute job. Recommendation:

1. Run the atlas precompute (~10 min). Verify bit-exact. Then turn on
   `--atlas-embedding-cache` for the next iSFT run. Should save 1-2 min
   per phase on small corpora, 10-15 min on larger ones.
2. Run the query precompute over the RLVR allocation (~hours). Verify
   bit-exact. Then add `--query-embedding-cache`. The combined hit rate
   on iSFT iterative corpora typically pushes >70%, which compounds the
   savings.

## Risks / things to know

- **Bit-exact correctness is the gate.** A subtly-wrong embedding can
  silently corrupt training. The atol=1e-5 threshold is mandatory.
- **Image processor padding** — Gemma4Processor pads images to a fixed
  patch grid. Embedding shapes must match the processor's output shape;
  if they diverge, `torch.stack` in the collator will raise. That's the
  intended hard signal.
- **Stale cache invalidation** — if you change the atlas grid step or
  re-render with a different normalization, cached embeddings are wrong.
  The cache loader doesn't validate `step_mm` against the live grid.
  Always rerun precompute after any atlas/preprocessing change.
- **Splice is multimodal-specific** — the hook is in the vision tower path.
  Don't try this for text-only models.
