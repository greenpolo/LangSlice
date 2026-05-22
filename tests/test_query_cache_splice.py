"""Tests for QueryEmbeddingCache wiring into the SFT splice path.

The atlas-cache integration is already covered in
``test_atlas_embedding_collate.py``. These tests confirm that:

1. The collator consults the query cache as a fallback when the atlas
   cache misses, treating both as path-keyed lookups behind the same
   sidecar machinery.
2. Atlas-cache hits still shadow query-cache hits on path collision so the
   ordering invariant the splice depends on holds.
3. The splice forward hook (already cache-agnostic) accepts mixed
   atlas+query embeddings — same flat-concat layout, no shape regression.
4. The disabled-cache path is byte-identical to the no-splice path.

The stubs mirror the ones in :mod:`test_atlas_embedding_collate` so the
two test files run on CPU / in CI without Gemma 4 weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from langslice_training.embeddings.cache import AtlasEmbeddingCache
from langslice_training.embeddings.query_cache import QueryEmbeddingCache, save_query_pair
from sft.collate import LangSliceCollator
from sft.render import RenderedExample, RenderMetadata

# ---------------------------------------------------------------------------
# Stubs (mirror test_atlas_embedding_collate)
# ---------------------------------------------------------------------------


def _make_metadata() -> RenderMetadata:
    return RenderMetadata(
        atlas_name="allen_mouse_25um",
        atlas_version="CCFv3",
        plane="coronal",
        subject_id="stub_subject",
        system_prompt_kind="single_slice",
    )


class _StubTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    def convert_tokens_to_ids(self, name):
        return -1


class _StubProcessor:
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


def _build_atlas_cache(
    tmp_path: Path, atlas: str, plane: str, basenames: list[str], hidden: int = 8,
) -> AtlasEmbeddingCache:
    out_dir = tmp_path / "atlas_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {b: torch.randn(1, hidden) for b in basenames}
    torch.save(
        {"atlas": atlas, "plane": plane, "embeddings": embeddings},
        out_dir / f"{atlas}_{plane}.pt",
    )
    return AtlasEmbeddingCache(out_dir)


def _build_query_cache(
    tmp_path: Path,
    pairs: list[tuple[str, str, dict[str, torch.Tensor]]],
) -> QueryEmbeddingCache:
    """Write per-(plane, dataset) cache files and return a QueryEmbeddingCache."""
    out_dir = tmp_path / "query_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for plane, dataset, embeddings in pairs:
        save_query_pair(out_dir, plane=plane, dataset=dataset, embeddings=embeddings)
    return QueryEmbeddingCache(out_dir)


# ---------------------------------------------------------------------------
# Sidecar emission: query cache alone
# ---------------------------------------------------------------------------


def test_query_only_cache_emits_sidecar(tmp_path):
    """Without atlas_cache but with query_cache, slice-path hits flow through."""
    path = "data/datasets/coronal/ad_bxd/subjA/sec1.jpg"
    qcache = _build_query_cache(
        tmp_path,
        [("coronal", "ad_bxd", {path: torch.full((4, 8), 7.0)})],
    )
    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=None,
        query_cache=qcache,
        enable_splice=True,
    )
    ex = _build_rendered_example([path])
    batch = collator([ex])
    assert batch["precomputed_image_mask"].tolist() == [True]
    assert batch["precomputed_cached_flat"].shape == (4, 8)
    assert batch["precomputed_cached_patch_counts"].tolist() == [4]
    counters = collator.cache_counters()
    assert counters["hits"] == 1
    assert counters["query_hits"] == 1
    assert counters["atlas_hits"] == 0


def test_query_only_cache_increments_misses_for_unknown_path(tmp_path):
    qcache = _build_query_cache(
        tmp_path,
        [(
            "coronal", "ad_bxd",
            {"data/datasets/coronal/ad_bxd/subjA/sec1.jpg": torch.zeros(4, 8)},
        )],
    )
    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=None,
        query_cache=qcache,
    )
    ex = _build_rendered_example([
        "data/datasets/coronal/ad_bxd/subjA/sec1.jpg",  # hit
        "data/datasets/coronal/ad_bxd/subjA/sec_missing.jpg",  # miss
        "queries/random.jpg",  # miss (not a canonical manifest path)
    ])
    collator([ex])
    counters = collator.cache_counters()
    assert counters["hits"] == 1
    assert counters["misses"] == 2
    assert counters["query_hits"] == 1
    assert counters["atlas_hits"] == 0


# ---------------------------------------------------------------------------
# Mixed atlas + query — sidecar layout
# ---------------------------------------------------------------------------


def test_mixed_atlas_and_query_sidecar(tmp_path):
    """A batch with atlas + query hits + a true miss emits a coherent sidecar."""
    atlas_cache = _build_atlas_cache(
        tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg"], hidden=8,
    )
    query_path = "data/datasets/coronal/ad_bxd/subjA/sec1.jpg"
    query_cache = _build_query_cache(
        tmp_path,
        [("coronal", "ad_bxd", {query_path: torch.full((4, 8), 3.0)})],
    )
    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=atlas_cache,
        query_cache=query_cache,
        enable_splice=True,
    )
    ex = _build_rendered_example([
        query_path,
        "atlas/allen_mouse_25um/coronal/9.60mm.jpg",
        "queries/none.jpg",  # neither cache covers this
    ])
    batch = collator([ex])
    mask = batch["precomputed_image_mask"].tolist()
    assert mask == [True, True, False]
    # cached_flat: 4 query patches + 1 atlas patch (1-D leading dim squeezed) = 5
    flat = batch["precomputed_cached_flat"]
    pc = batch["precomputed_cached_patch_counts"]
    assert flat.shape[0] == 5
    assert pc.tolist() == [4, 1]
    counters = collator.cache_counters()
    assert counters["hits"] == 2
    assert counters["misses"] == 1
    assert counters["atlas_hits"] == 1
    assert counters["query_hits"] == 1


def test_atlas_takes_precedence_on_collision(tmp_path):
    """If a path is in both caches, atlas wins; counters log the atlas hit only."""
    # Pathological case — both caches map the same path, but atlas is canonical.
    # We can't actually get an atlas-style path into the query cache via the
    # public save helper (its files key by manifest path), so test the
    # precedence logic by stubbing the query cache to lie about a collision.
    atlas_cache = _build_atlas_cache(
        tmp_path, "allen_mouse_25um", "coronal", ["9.60mm.jpg"], hidden=8,
    )

    class _LyingQueryCache:
        """Stub that claims a hit for the atlas path to test precedence."""

        def lookup_by_path(self, path):
            if path.endswith("9.60mm.jpg"):
                return torch.full((2, 8), 99.0)
            return None

    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=atlas_cache,
        query_cache=_LyingQueryCache(),
        enable_splice=True,
    )
    ex = _build_rendered_example(["atlas/allen_mouse_25um/coronal/9.60mm.jpg"])
    batch = collator([ex])
    flat = batch["precomputed_cached_flat"]
    # Atlas cache emitted shape (1, 8) → squeezed to (8,) per image. The lying
    # query cache would have given (2, 8). If atlas wins, flat has 1 row; if
    # query wins, flat has 2.
    assert flat.shape[0] == 1, "atlas cache must shadow query cache on collision"
    counters = collator.cache_counters()
    assert counters["atlas_hits"] == 1
    assert counters["query_hits"] == 0


# ---------------------------------------------------------------------------
# Splice-disabled / no-cache path: byte-identical to existing behavior
# ---------------------------------------------------------------------------


def test_no_caches_no_sidecar(tmp_path):
    """When neither cache is provided, no sidecar keys land in the batch."""
    collator = LangSliceCollator(processor=_StubProcessor(), max_seq_length=64)
    ex = _build_rendered_example(["queries/x.jpg", "atlas/whatever/coronal/0.00mm.jpg"])
    batch = collator([ex])
    assert "precomputed_image_mask" not in batch
    assert "precomputed_cached_flat" not in batch
    assert "precomputed_cached_patch_counts" not in batch


def test_query_cache_without_splice_flag_emits_no_sidecar(tmp_path):
    """Cache present but enable_splice=False → no sidecar emitted (still useful for measurement)."""
    qcache = _build_query_cache(
        tmp_path,
        [(
            "coronal", "ad_bxd",
            {"data/datasets/coronal/ad_bxd/subjA/sec1.jpg": torch.zeros(4, 8)},
        )],
    )
    collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=None,
        query_cache=qcache,
        enable_splice=False,
    )
    ex = _build_rendered_example(["data/datasets/coronal/ad_bxd/subjA/sec1.jpg"])
    batch = collator([ex])
    assert "precomputed_image_mask" not in batch
    # Hit counter still increments — the cache is consulted for measurement.
    counters = collator.cache_counters()
    assert counters["hits"] == 1
    assert counters["query_hits"] == 1


def test_all_miss_cache_byte_identical_to_no_cache(tmp_path):
    """Gate A (byte-identical): when every lookup misses, splice-enabled output
    matches the no-cache batch exactly.

    This is the corpus-vs-cache mismatch case (e.g. iSFT-staged paths against a
    manifest-keyed query cache): the cache misses every image, the splice
    branch never emits a sidecar, and the collator output must equal the
    no-cache code path's output token-for-token.
    """
    # Build a query cache that won't match any of the example's paths.
    qcache = _build_query_cache(
        tmp_path,
        [(
            "coronal", "ad_bxd",
            {"data/datasets/coronal/ad_bxd/subjA/different_image.jpg": torch.zeros(4, 8)},
        )],
    )
    paths = [
        "queries/staged_001.jpg",
        "atlas/whatever/coronal/0.00mm.jpg",
    ]

    # No-cache reference run.
    ref_collator = LangSliceCollator(processor=_StubProcessor(), max_seq_length=64)
    ref_batch = ref_collator([_build_rendered_example(paths)])

    # Cache-present, all-miss run.
    cache_collator = LangSliceCollator(
        processor=_StubProcessor(),
        max_seq_length=64,
        atlas_cache=None,
        query_cache=qcache,
        enable_splice=True,
    )
    cache_batch = cache_collator([_build_rendered_example(paths)])

    # Same keys, same tensor values — sidecar keys must be absent.
    assert set(ref_batch.keys()) == set(cache_batch.keys()), (
        f"key sets differ: ref={set(ref_batch.keys())} cache={set(cache_batch.keys())}"
    )
    assert "precomputed_image_mask" not in cache_batch
    assert "precomputed_cached_flat" not in cache_batch
    assert "precomputed_cached_patch_counts" not in cache_batch
    for k in ref_batch:
        assert torch.equal(ref_batch[k], cache_batch[k]), f"tensor mismatch at key {k!r}"

    assert cache_collator.cache_counters()["misses"] == len(paths)
    assert cache_collator.cache_counters()["hits"] == 0


# ---------------------------------------------------------------------------
# Splice forward hook: shape-invariant across atlas-only, query-only, mixed
# ---------------------------------------------------------------------------


def test_splice_hook_accepts_mixed_sidecars():
    """The splice forward hook must consume mixed atlas+query sidecars without complaint.

    Builds a fake Gemma inner that mirrors the real shape contract and feeds
    it a hand-crafted (mask, flat, pc) tuple where each cached slot has a
    different per-image patch count — simulating the heterogeneous mix that
    atlas (small) + query (large) caches produce in practice.
    """
    import torch.nn as nn
    from langslice_training.embeddings.splice import install_atlas_splice

    class _FakeOut:
        def __init__(self, lhs, pool):
            self.last_hidden_state = lhs
            self.pooler_output = pool

    class _FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Identity()

            class _Embed(nn.Module):
                def __call__(self, *, inputs_embeds):
                    return inputs_embeds * 2

            self.embed_vision = _Embed()
            self.calls = []

        def get_image_features(
            self, pixel_values, image_position_ids=None, **kwargs,
        ):
            self.calls.append(int(pixel_values.shape[0]))
            real = (image_position_ids >= 0).all(dim=-1)
            per = real.sum(dim=-1)
            total = int(per.sum().item())
            rows = torch.arange(total, dtype=torch.float32).view(total, 1).expand(total, 4).clone()
            return _FakeOut(rows, rows * 2)

        def forward(self, **kwargs):
            return self.get_image_features(
                kwargs.get("pixel_values"),
                kwargs.get("image_position_ids"),
            )

    model = _FakeInner()
    install_atlas_splice(model)

    # 3 images: image 0 cached (query — 5 patches), image 1 cached (atlas — 1 patch),
    # image 2 uncached (3 patches, runs live)
    pv = torch.zeros(3, 8, 8)
    pid = torch.full((3, 8, 2), -1, dtype=torch.long)
    pid[0, :5] = 0
    pid[1, :1] = 0
    pid[2, :3] = 0
    mask = torch.tensor([True, True, False])
    # Cached flat: image-0 has 5 patches of value 100, image-1 has 1 patch of value 50.
    cached_flat = torch.cat([
        torch.full((5, 4), 100.0),
        torch.full((1, 4), 50.0),
    ], dim=0)
    cached_pc = torch.tensor([5, 1], dtype=torch.long)

    out = model(
        pixel_values=pv,
        image_position_ids=pid,
        precomputed_image_mask=mask,
        precomputed_cached_flat=cached_flat,
        precomputed_cached_patch_counts=cached_pc,
    )
    # Original called exactly once with 1 uncached image
    assert model.calls == [1]
    # Output flat layout: 5 (cached image 0) + 1 (cached image 1) + 3 (live image 2) = 9
    assert out.last_hidden_state.shape == (9, 4)
    # First 5 rows should match the cached image 0 (value 100)
    assert torch.allclose(out.last_hidden_state[0:5], torch.full((5, 4), 100.0))
    # Row 5 should match cached image 1 (value 50)
    assert torch.allclose(out.last_hidden_state[5:6], torch.full((1, 4), 50.0))
    # Rows 6-8 are live (fake encoder generates rows of running index 0, 1, 2)
    assert torch.allclose(out.last_hidden_state[6], torch.zeros(4))
    assert torch.allclose(out.last_hidden_state[7], torch.ones(4))
    assert torch.allclose(out.last_hidden_state[8], torch.full((4,), 2.0))


# ---------------------------------------------------------------------------
# CLI parsing: --query-embedding-cache on train_sft
# ---------------------------------------------------------------------------


def test_train_sft_query_cache_flag_parses(tmp_path):
    """The CLI surfaces --query-embedding-cache as an absent-by-default Path arg."""
    from sft.train_sft import _parse_args

    dataset = tmp_path / "ds.jsonl"
    dataset.write_text("", encoding="utf-8")
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[sft]\n[lora]\n[data]\nholdout_fraction = 0.1\n", encoding="utf-8")
    out = tmp_path / "out"
    qcache = tmp_path / "qcache"

    # Without the flag, default should be None
    args = _parse_args([
        "--config", str(cfg),
        "--dataset", str(dataset),
        "--output-dir", str(out),
    ])
    assert args.query_embedding_cache is None

    # With the flag, it parses to a Path
    args = _parse_args([
        "--config", str(cfg),
        "--dataset", str(dataset),
        "--output-dir", str(out),
        "--query-embedding-cache", str(qcache),
    ])
    assert args.query_embedding_cache == qcache


def test_train_sft_query_cache_help_advertises_flag(tmp_path):
    """--help output mentions the new flag so operators can discover it."""
    import io
    import sys as _sys

    from sft.train_sft import _parse_args as parse_train_sft

    buf = io.StringIO()
    old = _sys.stdout
    _sys.stdout = buf
    try:
        with pytest.raises(SystemExit):
            parse_train_sft(["--help"])
    finally:
        _sys.stdout = old
    assert "--query-embedding-cache" in buf.getvalue()
