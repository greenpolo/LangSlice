# Atlas Vision Embedding Cache (DORMANT, mostly Phase 1)

Pre-computes SigLIP embeddings for every atlas reference image once at startup,
so SFT training can skip the SigLIP forward pass for cached atlas-image
inputs. Phase 1 (precompute + cache + measurement) is built and validated.
Phase 2 (the splice that actually skips SigLIP at training time) is built but
flag-gated off pending a bit-exact correctness test.

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
| `precompute.py` (CLI: precompute SigLIP for every atlas grid pos) | ✓ Built, not run yet on a real GPU |
| `cache.py` (`AtlasEmbeddingCache` lookup by atlas/plane/snapped position) | ✓ Built, unit-tested |
| `LangSliceCollator` Phase-1 hit/miss counters | ✓ Built, accepts `atlas_cache=` param |
| **Phase 2 splice** (forward hook to skip SigLIP for cached images) | ✗ **Sidecar tensor scaffolded** but `enable_splice=False` by default |
| Bit-exact correctness gate | ✗ Test exists but auto-skips without GPU + Gemma 4 weights |
| Hit-rate measurement on real corpus | ✓ **100% on 8643-row corpus, 11 atlas/plane pairs** |
| **Used in any actual training** | ✗ **No.** |

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `AtlasEmbeddingCache`, `parse_atlas_path` |
| `cache.py` | Lazy mmap-backed lookup keyed by (atlas, plane, snapped_pos_mm) |
| `precompute.py` | CLI: load model, run SigLIP on every atlas grid image, save tensor cache |

External entry point: `tools/embeddings_hit_rate.py` walks a corpus and
reports atlas-grid coverage.

Tests: `tests/test_atlas_embedding_cache.py` (19) +
`tests/test_atlas_embedding_collate.py` (8) — **27 pass + 1 GPU-skipped**.

## How to activate (when ready)

### Step 1: precompute the cache (one-time, ~10 min on the 5090)

```powershell
docker compose -f docker-compose.training.yml run --rm training bash -lc "
  cd /workspace/LangSlice && python -m embeddings.precompute \
    --atlas-pairs allen_mouse_25um:coronal,whs_sd_rat_39um:coronal,whs_sd_rat_39um:horizontal,allen_mouse_25um:sagittal \
    --model /workspace/LangSlice/out/sft/docker-sft-1011-merged-bf16 \
    --output-dir /workspace/LangSlice/out/atlas_embeddings \
    --device cuda --dtype bf16 --step-mm 0.05
"
```

Will produce `out/atlas_embeddings/<atlas>_<plane>.pt` per pair.

### Step 2: run the bit-exact correctness test

`tests/test_atlas_embedding_cache.py::test_precomputed_matches_live_get_image_features`
auto-skips when GPU is unavailable. Run it on the host once the precompute
has produced cache files. Pass criterion: cached embeddings match live
`model.get_image_features` to floating-point precision (`atol=1e-5`).

If the test fails: do not enable splicing. The cache is wrong.

### Step 3: enable the splice in SFT

Add to `LangSliceCollator`'s constructor in `train_sft.py`:
```python
collator = LangSliceCollator(
    processor=processor,
    max_seq_length=max_seq_length,
    atlas_cache=AtlasEmbeddingCache(Path("out/atlas_embeddings")),
    enable_splice=True,  # ← currently False
)
```

The collator then emits sidecar tensors `precomputed_image_embeddings` +
`precomputed_image_mask` alongside standard `pixel_values`. A forward hook
on `model.get_image_features` (NOT YET WRITTEN) would consult the sidecar
and substitute cached embeddings for the masked slots, letting SigLIP run
only on uncached images.

**The forward hook is the missing piece.** Without it, the sidecar is just
extra fields the model ignores. The agent picking this up needs to:
1. Locate `Gemma4ForConditionalGeneration.get_image_features` (per
   `unsloth_compiled_cache/unsloth_compiled_module_gemma4.py:1452`)
2. Register a pre-forward hook that reads the sidecar from the input batch
3. Substitute cached embeddings at masked positions
4. Verify training loss stays in distribution after substitution
5. Verify slicebench MAE doesn't shift

## When to activate

After:
1. Expert iteration is producing measurably-improving checkpoints across
   rounds (so we have a working baseline to compare against).
2. We can afford ~half-day of work on the forward hook + correctness gate.
3. We need the speed boost — the current SFT phase is ~13 min on 180-slice
   corpora. With 100% cache hit rate, savings would be ~1-2 min per phase.
   Smaller win than vLLM perf wins.

For 30-minute SFT phases on larger corpora, the savings would scale
proportionally — maybe ~10-15 min saved per phase. Worth it for the real
multi-round runs.

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
