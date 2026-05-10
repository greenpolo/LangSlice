"""Tests for atlas-embedding integration with LangSliceCollator.

Phase 1: hit/miss counters increment correctly when the collator sees cached
vs uncached image paths.

Phase 2: with the splice flag enabled, the collator output exposes
``precomputed_image_embeddings`` + ``precomputed_image_mask`` for cached
slots so the model-side hook can intercept before SigLIP runs.

These tests use stub processors and stub examples — no Gemma 4 weights or
real chat-template machinery required, so they run on CPU / in CI alongside
the existing collator unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from embeddings.cache import AtlasEmbeddingCache
from sft.collate import LangSliceCollator
from sft.render import RenderedExample, RenderMetadata


def _make_metadata() -> RenderMetadata:
    return RenderMetadata(
        atlas_name="allen_mouse_25um",
        atlas_version="CCFv3",
        plane="coronal",
        subject_id="stub_subject",
        system_prompt_kind="single_slice",
    )


def _build_cache(tmp_path: Path, atlas: str, plane: str, basenames: list[str]) -> Path:
    out_dir = tmp_path / "atlas_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {b: torch.randn(1, 8) for b in basenames}
    torch.save(
        {
            "atlas": atlas,
            "plane": plane,
            "embeddings": embeddings,
        },
        out_dir / f"{atlas}_{plane}.pt",
    )
    return out_dir


class _StubTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    def convert_tokens_to_ids(self, name):
        return -1


class _StubProcessor:
    """Minimal processor stub mirroring the parts of HuggingFace's API the collator uses.

    Returns deterministic short input_ids per example with a dummy assistant
    mask covering the second half. ``pixel_values`` is a (n_images, 3, 4, 4)
    tensor so _pad_batch's image-tensor concat path runs.
    """

    tokenizer = _StubTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        n_imgs = sum(
            1
            for m in messages
            for block in (m.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "image"
        )
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attn = torch.ones_like(ids)
        # First half = system/user, second half = assistant — gives a non-trivial label mask.
        amask = torch.zeros_like(ids)
        amask[:, 4:] = 1
        out = {
            "input_ids": ids,
            "attention_mask": attn,
            "assistant_masks": amask,
        }
        if n_imgs > 0:
            out["pixel_values"] = torch.zeros(n_imgs, 3, 4, 4)
        return out


def _build_rendered_example(image_paths: list[str]) -> RenderedExample:
    """RenderedExample with a placeholder messages list whose image-block count matches paths."""
    user_content = [{"type": "image", "image": object()} for _ in image_paths]
    user_content.append({"type": "text", "text": "Estimate the position of this slice."})
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": "OK"}]},
    ]
    return RenderedExample(
        messages=messages,
        tools=[],
        metadata=_make_metadata(),
        image_paths=image_paths,
    )


# ---------------------------------------------------------------------------
# Phase 1 — hit/miss counters
# ---------------------------------------------------------------------------


def test_collator_no_cache_returns_zero_hit_rate():
    collator = LangSliceCollator(processor=_StubProcessor(), max_seq_length=64)
    ex = _build_rendered_example(["queries/some.jpg"])
    collator([ex])
    assert collator.cache_hit_rate() == 0.0
    assert collator.cache_counters() == {"hits": 0, "misses": 0}


def test_collator_with_cache_increments_hits(tmp_path):
    cache_dir = _build_cache(tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg", "9.80mm.jpg"])
    cache = AtlasEmbeddingCache(cache_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64, atlas_cache=cache,
    )
    ex = _build_rendered_example([
        "atlas/allen_mouse_25um/coronal/9.60mm.jpg",
        "atlas/allen_mouse_25um/coronal/9.80mm.jpg",
    ])
    collator([ex])
    counters = collator.cache_counters()
    assert counters["hits"] == 2
    assert counters["misses"] == 0
    assert collator.cache_hit_rate() == 1.0


def test_collator_with_cache_increments_misses(tmp_path):
    cache_dir = _build_cache(tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg"])
    cache = AtlasEmbeddingCache(cache_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64, atlas_cache=cache,
    )
    ex = _build_rendered_example([
        "queries/non_atlas.jpg",                                       # miss (not atlas)
        "atlas/allen_mouse_25um/coronal/9.60mm.jpg",                   # hit
        "atlas/allen_mouse_25um/coronal/3.00mm.jpg",                   # miss (not in cache)
        "atlas/whs_sd_rat_39um/coronal/2.00mm.jpg",                    # miss (atlas not cached)
    ])
    collator([ex])
    counters = collator.cache_counters()
    assert counters["hits"] == 1
    assert counters["misses"] == 3
    assert collator.cache_hit_rate() == pytest.approx(1 / 4)


def test_collator_hit_rate_accumulates_across_batches(tmp_path):
    cache_dir = _build_cache(tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg"])
    cache = AtlasEmbeddingCache(cache_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64, atlas_cache=cache,
    )
    hit_path = "atlas/allen_mouse_25um/coronal/9.60mm.jpg"
    miss_path = "queries/none.jpg"
    collator([_build_rendered_example([hit_path, miss_path])])
    collator([_build_rendered_example([miss_path, miss_path, miss_path])])
    counters = collator.cache_counters()
    assert counters["hits"] == 1
    assert counters["misses"] == 4
    assert collator.cache_hit_rate() == pytest.approx(1 / 5)


# ---------------------------------------------------------------------------
# Phase 2 — splice-mode sidecar
# ---------------------------------------------------------------------------


def test_splice_disabled_by_default_omits_sidecar(tmp_path):
    cache_dir = _build_cache(tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg"])
    cache = AtlasEmbeddingCache(cache_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64, atlas_cache=cache,
    )
    ex = _build_rendered_example(["atlas/allen_mouse_25um/coronal/9.60mm.jpg"])
    batch = collator([ex])
    assert "precomputed_image_embeddings" not in batch
    assert "precomputed_image_mask" not in batch


def test_splice_enabled_emits_sidecar(tmp_path):
    cache_dir = _build_cache(tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg", "9.80mm.jpg"])
    cache = AtlasEmbeddingCache(cache_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64,
        atlas_cache=cache, enable_splice=True,
    )
    ex = _build_rendered_example([
        "atlas/allen_mouse_25um/coronal/9.60mm.jpg",
        "queries/no_match.jpg",
        "atlas/allen_mouse_25um/coronal/9.80mm.jpg",
    ])
    batch = collator([ex])
    mask = batch["precomputed_image_mask"]
    flat = batch["precomputed_cached_flat"]
    pc = batch["precomputed_cached_patch_counts"]
    assert mask.tolist() == [True, False, True]
    # Cache entries here are 2D (1, 8) shaped — the collator's leading-1
    # squeeze drops the 1, leaving per-image (8,) → flat (2*8 = ?, ?)…
    # Actually each cached entry is 2D shape (1, 8) → squeezed to (8,). The
    # flat concat then has shape (2 * 1, 8) where the 1 means "this entry
    # represents 1 patch" (a degenerate test fixture). The patch counts
    # tensor records that.
    assert flat.shape == (2, 8)  # 2 cached images × 1 patch each (test fixture)
    assert pc.tolist() == [1, 1]


def test_splice_requires_cache():
    with pytest.raises(ValueError, match="enable_splice=True requires atlas_cache"):
        LangSliceCollator(
            processor=_StubProcessor(), max_seq_length=64,
            atlas_cache=None, enable_splice=True,
        )


def test_splice_squeezes_leading_batch_dim_from_cache(tmp_path):
    """Real SigLIP cache files have shape ``(1, num_patches, hidden)`` per
    qkey (the processor's batch=1 dim). The collator must peel that leading 1
    so the stacked sidecar has shape ``(N_cached, num_patches, hidden)`` —
    otherwise the splice's ``out[mask] = cached`` indexing fails with a
    shape-mismatch RuntimeError.
    """
    out_dir = tmp_path / "atlas_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Real precompute used to emit (1, 266, 768) per cache entry — the leading
    # "1" was a processor batch dim that broke `out[mask] = cached`. The fix
    # is in the collator's per-image squeeze. This test guards against that
    # regression even though the current precompute already squeezes at save
    # time.
    embeddings = {
        "9.60mm.jpg": torch.randn(1, 16, 32),
        "9.80mm.jpg": torch.randn(1, 16, 32),
    }
    torch.save(
        {
            "atlas": "allen_mouse_25um",
            "plane": "coronal",
            "embeddings": embeddings,
        },
        out_dir / "allen_mouse_25um_coronal.pt",
    )
    cache = AtlasEmbeddingCache(out_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64,
        atlas_cache=cache, enable_splice=True,
    )
    ex = _build_rendered_example([
        "atlas/allen_mouse_25um/coronal/9.60mm.jpg",
        "atlas/allen_mouse_25um/coronal/9.80mm.jpg",
    ])
    batch = collator([ex])
    flat = batch["precomputed_cached_flat"]
    pc = batch["precomputed_cached_patch_counts"]
    # After the per-image leading-1 squeeze, each cached entry is (16, 32);
    # concatenating two of them gives (32, 32). Patch counts record [16, 16].
    assert flat.shape == (32, 32)
    assert pc.tolist() == [16, 16]


def test_splice_no_cache_hits_omits_sidecar(tmp_path):
    """All-miss batch with splice enabled emits no sidecar (forward hook would no-op)."""
    cache_dir = _build_cache(tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg"])
    cache = AtlasEmbeddingCache(cache_dir)
    collator = LangSliceCollator(
        processor=_StubProcessor(), max_seq_length=64,
        atlas_cache=cache, enable_splice=True,
    )
    ex = _build_rendered_example(["queries/none1.jpg", "queries/none2.jpg"])
    batch = collator([ex])
    assert "precomputed_image_embeddings" not in batch
    assert "precomputed_image_mask" not in batch
