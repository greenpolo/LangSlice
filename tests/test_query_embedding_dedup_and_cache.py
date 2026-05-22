"""Tests for two related wins on the single-turn GRPO trainer:

1. **Within-step query dedup**: with ``num_generations=N`` GRPO repeats each
   prompt N times in one ``generation_batch``. The query (first) image_path
   per row is therefore identical across the N rollouts. The sidecar build
   should detect duplicates and encode each unique query ONCE per batch,
   marking the duplicates as cached so the splice forward skips re-encoding.

2. **Persistent query embedding cache**: a sibling of the atlas embedding
   cache, keyed by the full repo-relative query image_path. Loaded at trainer
   startup; lookups merge with atlas-cache hits in the sidecar build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from langslice_training.rl.single_turn import train_grpo as tg

# ---------------------------------------------------------------------------
# Stub helpers (mirror existing test_single_turn_train_grpo style)
# ---------------------------------------------------------------------------


class _StubCache:
    """Path-keyed cache stub (mirrors AtlasEmbeddingCache.lookup_by_path)."""

    def __init__(self, hits: dict[str, torch.Tensor]) -> None:
        self._hits = dict(hits)

    def lookup_by_path(self, path: str) -> torch.Tensor | None:
        emb = self._hits.get(path)
        if emb is None:
            return None
        return emb.clone()


class _CacheChain:
    """Two-tier lookup: atlas cache wins on collision, query cache fills the rest.

    Mirrors what train_grpo wires when both ``--atlas-embedding-cache`` and
    ``--query-embedding-cache`` are set.
    """

    def __init__(self, atlas: _StubCache | None, query: _StubCache | None) -> None:
        self._atlas = atlas
        self._query = query

    def lookup_by_path(self, path: str) -> torch.Tensor | None:
        if self._atlas is not None:
            hit = self._atlas.lookup_by_path(path)
            if hit is not None:
                return hit
        if self._query is not None:
            return self._query.lookup_by_path(path)
        return None


def _make_pid(per_image_patches: list[int], max_patches: int | None = None) -> torch.Tensor:
    """Build an image_position_ids tensor mirroring Gemma 4's ``(N, max_p, 2)`` shape.

    Real patches use ``(>=0, >=0)``; padding uses ``(-1, -1)``.
    """
    n = len(per_image_patches)
    if max_patches is None:
        max_patches = max(per_image_patches) if per_image_patches else 1
    pid = torch.full((n, max_patches, 2), -1, dtype=torch.long)
    for i, count in enumerate(per_image_patches):
        for j in range(count):
            pid[i, j, 0] = j
            pid[i, j, 1] = j
    return pid


def _fake_encoder_factory(hidden: int = 4):
    """Returns an encoder callable + a counter recording (pv_shape, pid_shape) per call.

    The encoder mirrors the splice's expected return: a namespace with
    ``last_hidden_state`` of shape ``(sum_real_patches, hidden)``. Each row
    is filled with the row index so callers can assert per-image slices.
    """
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class _Out:
        def __init__(self, lhs: torch.Tensor) -> None:
            self.last_hidden_state = lhs

    def _encode(pv: torch.Tensor, pid: torch.Tensor) -> _Out:
        calls.append((tuple(pv.shape), tuple(pid.shape)))
        real = (pid >= 0).all(dim=-1)
        per = real.sum(dim=-1).tolist()
        total = int(sum(per))
        # row j filled with value (call_idx * 1000 + j) so multi-call tests
        # can keep encodings distinguishable
        offset = sum(c[0][0] for c in calls[:-1])  # crude unique base per call
        rows = torch.arange(total, dtype=torch.float32) + 1000 * (len(calls) - 1) + offset
        lhs = rows.view(total, 1).expand(total, hidden).clone()
        return _Out(lhs)

    return _encode, calls


# ---------------------------------------------------------------------------
# WIN #1: within-step query dedup
# ---------------------------------------------------------------------------


def test_within_step_query_dedup_single_unique() -> None:
    """G=4 copies of the same row → encoder runs once; mask = [True]*8 (4 query + 4 atlas).

    Atlas paths in this test are also identical across rows (since same row
    repeated), so without disk cache they'd ALSO dedupe down to 1 unique
    encode. Total: 2 unique images (1 query + 1 atlas), 1 encode call (since
    we batch all uniques into one encoder call).
    """
    G = 4
    query_path = "queries/brain_a.jpg"
    atlas_path = "atlas/x/coronal/2.00mm.jpg"
    images_per_row = ["pil_q_a", "pil_a_2"]
    pixel_values = torch.zeros(G * 2, 3, 4, 4)  # G rows × 2 imgs each
    pid = _make_pid([3] * (G * 2))  # 3 real patches each
    encode_fn, calls = _fake_encoder_factory(hidden=4)

    sidecars = tg._build_sidecars(
        images=[images_per_row] * G,
        paths_per_row=[[query_path, atlas_path]] * G,
        cache=_StubCache({}),  # no disk hits
        pixel_values=pixel_values,
        image_position_ids=pid,
        encode_fn=encode_fn,
    )
    assert sidecars is not None
    mask, cached_flat, cached_pc = sidecars
    # Every slot is now "cached" because we pre-encoded the uniques.
    assert mask.tolist() == [True] * (G * 2)
    # Encoder ran exactly ONCE on the 2 unique images (batched together).
    assert len(calls) == 1
    pv_shape, pid_shape = calls[0]
    assert pv_shape[0] == 2  # 2 unique images submitted
    assert pid_shape[0] == 2
    # cached_flat row count = G * 2 images * 3 patches = 24
    assert cached_flat.shape[0] == G * 2 * 3
    assert cached_pc.tolist() == [3] * (G * 2)


def test_within_step_query_dedup_multiple_unique() -> None:
    """2 unique queries × G=2 each + identical atlas in all 4 rows → encoder runs on 3 uniques."""
    q1 = "queries/brain_a.jpg"
    q2 = "queries/brain_b.jpg"
    atlas = "atlas/x/coronal/3.00mm.jpg"
    paths_per_row = [
        [q1, atlas],
        [q1, atlas],
        [q2, atlas],
        [q2, atlas],
    ]
    images_per_row = [["pil_q1", "pil_atlas"]] * 2 + [["pil_q2", "pil_atlas"]] * 2
    pixel_values = torch.zeros(8, 3, 4, 4)
    pid = _make_pid([2] * 8)
    encode_fn, calls = _fake_encoder_factory(hidden=4)
    sidecars = tg._build_sidecars(
        images=images_per_row,
        paths_per_row=paths_per_row,
        cache=_StubCache({}),
        pixel_values=pixel_values,
        image_position_ids=pid,
        encode_fn=encode_fn,
    )
    assert sidecars is not None
    mask, cached_flat, cached_pc = sidecars
    # All 8 slots cached (2 unique queries + 1 unique atlas pre-encoded once each)
    assert mask.tolist() == [True] * 8
    assert len(calls) == 1
    pv_shape, _ = calls[0]
    # 3 unique images encoded
    assert pv_shape[0] == 3
    assert cached_flat.shape[0] == 8 * 2  # 8 slots × 2 patches each
    assert cached_pc.tolist() == [2] * 8


def test_within_step_dedup_disabled_when_encoder_missing() -> None:
    """Without ``encode_fn``, dedup is impossible; behaviour falls back to disk-cache-only."""
    paths = [["queries/q.jpg", "atlas/x/coronal/2.00mm.jpg"]] * 3
    sidecars = tg._build_sidecars(
        images=[["pil_q", "pil_a"]] * 3,
        paths_per_row=paths,
        cache=_StubCache({}),  # no disk hits → no sidecar
        # pixel_values + image_position_ids + encode_fn omitted
    )
    # No disk hits, no dedup possible → returns None (no sidecar emitted)
    assert sidecars is None


def test_within_step_dedup_no_duplicates_skips_pre_encoding() -> None:
    """Single-row batch (no duplicates) shouldn't pre-encode — saves zero work, adds latency."""
    paths = [["queries/q1.jpg", "atlas/x/coronal/2.00mm.jpg"]]
    pv = torch.zeros(2, 3, 4, 4)
    pid = _make_pid([1, 1])
    encode_fn, calls = _fake_encoder_factory(hidden=4)
    sidecars = tg._build_sidecars(
        images=[["pil_q", "pil_a"]],
        paths_per_row=paths,
        cache=_StubCache({}),  # no disk hits
        pixel_values=pv,
        image_position_ids=pid,
        encode_fn=encode_fn,
    )
    # No duplicates → pre-encoding is wasted work; sidecar build returns None
    # (falls through to splice forward's normal SigLIP path).
    assert sidecars is None
    assert len(calls) == 0


def test_within_step_dedup_disk_cache_takes_precedence_for_first_occurrence() -> None:
    """If the first occurrence is a disk-cache hit, it doesn't go through the encoder.

    The disk-cached embedding is reused for all duplicates, so the encoder
    only runs on uncached uniques (here: zero, since the only path is cached).
    """
    G = 3
    cached_path = "queries/brain_a.jpg"
    disk_emb = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    cache = _StubCache({cached_path: disk_emb})
    pv = torch.zeros(G, 3, 4, 4)
    pid = _make_pid([2] * G)
    encode_fn, calls = _fake_encoder_factory(hidden=4)
    sidecars = tg._build_sidecars(
        images=[["pil_q"]] * G,
        paths_per_row=[[cached_path]] * G,
        cache=cache,
        pixel_values=pv,
        image_position_ids=pid,
        encode_fn=encode_fn,
    )
    assert sidecars is not None
    mask, cached_flat, cached_pc = sidecars
    assert mask.tolist() == [True] * G
    # Disk cache had the path → encoder never invoked.
    assert len(calls) == 0
    # cached_flat = disk embedding repeated 3×
    assert cached_flat.shape == (G * 2, 4)
    assert cached_pc.tolist() == [2] * G


# ---------------------------------------------------------------------------
# WIN #2: persistent query embedding cache
# ---------------------------------------------------------------------------


def test_query_cache_save_load_roundtrip(tmp_path: Path) -> None:
    """Precompute writes a per-(plane, dataset) .pt file; reload looks it up."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache, save_query_pair

    plane = "coronal"
    dataset = "ad_bxd"
    embeddings = {
        "data/datasets/coronal/ad_bxd/subjA/sec1.jpg": torch.randn(4, 8),
        "data/datasets/coronal/ad_bxd/subjA/sec2.jpg": torch.randn(4, 8),
    }
    out_path = save_query_pair(
        tmp_path, plane=plane, dataset=dataset, embeddings=embeddings,
    )
    assert out_path.exists()

    cache = QueryEmbeddingCache(tmp_path)
    assert (plane, dataset) in cache.pairs()
    for path, expected in embeddings.items():
        got = cache.lookup_by_path(path)
        assert got is not None
        assert torch.equal(got, expected)


def test_query_cache_unknown_path_returns_none(tmp_path: Path) -> None:
    """Lookup for a path not in any cache file returns None (no exception)."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache

    cache = QueryEmbeddingCache(tmp_path)  # empty dir
    assert cache.lookup_by_path("queries/nope.jpg") is None
    assert cache.pairs() == []


def test_query_cache_separates_planes_and_datasets(tmp_path: Path) -> None:
    """(plane, dataset) pairs map to separate cache files; lookups are isolated."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache, save_query_pair

    save_query_pair(
        tmp_path, plane="coronal", dataset="ad_bxd",
        embeddings={"data/datasets/coronal/ad_bxd/x.jpg": torch.ones(2, 4)},
    )
    save_query_pair(
        tmp_path, plane="sagittal", dataset="allen_ish_sag",
        embeddings={"data/datasets/sagittal/allen_ish_sag/y.jpg": torch.zeros(2, 4)},
    )
    cache = QueryEmbeddingCache(tmp_path)
    assert ("coronal", "ad_bxd") in cache.pairs()
    assert ("sagittal", "allen_ish_sag") in cache.pairs()

    a = cache.lookup_by_path("data/datasets/coronal/ad_bxd/x.jpg")
    b = cache.lookup_by_path("data/datasets/sagittal/allen_ish_sag/y.jpg")
    assert a is not None and b is not None
    assert torch.equal(a, torch.ones(2, 4))
    assert torch.equal(b, torch.zeros(2, 4))


def test_query_cache_returns_clone(tmp_path: Path) -> None:
    """Mutating returned tensor must not corrupt the cache."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache, save_query_pair

    p = "data/datasets/coronal/ad_bxd/x.jpg"
    save_query_pair(
        tmp_path, plane="coronal", dataset="ad_bxd",
        embeddings={p: torch.zeros(2, 4)},
    )
    cache = QueryEmbeddingCache(tmp_path)
    first = cache.lookup_by_path(p)
    assert first is not None
    first.fill_(99.0)
    second = cache.lookup_by_path(p)
    assert second is not None
    assert (second == 0).all().item()


def test_query_cache_unions_with_atlas(tmp_path: Path) -> None:
    """Combined lookup: atlas hit takes precedence on collision; otherwise query
    fills in. Mirrors what the trainer wires when both flags are passed."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache, save_query_pair

    # Atlas cache (stub via _StubCache for isolation).
    atlas_emb = torch.full((2, 4), 1.0)
    atlas_cache = _StubCache({"atlas/x/coronal/2.00mm.jpg": atlas_emb})

    # Query cache on disk.
    save_query_pair(
        tmp_path, plane="coronal", dataset="ad_bxd",
        embeddings={"data/datasets/coronal/ad_bxd/x.jpg": torch.full((2, 4), 2.0)},
    )
    query_cache = QueryEmbeddingCache(tmp_path)

    chain = _CacheChain(atlas=atlas_cache, query=query_cache)

    # Atlas path → atlas hit.
    a = chain.lookup_by_path("atlas/x/coronal/2.00mm.jpg")
    assert a is not None and (a == 1.0).all().item()

    # Query path → query hit.
    q = chain.lookup_by_path("data/datasets/coronal/ad_bxd/x.jpg")
    assert q is not None and (q == 2.0).all().item()

    # Unknown path → None.
    assert chain.lookup_by_path("queries/random_unknown.jpg") is None


def test_query_cache_atlas_takes_precedence_on_collision(tmp_path: Path) -> None:
    """If both caches happen to contain the same key, atlas wins (deterministic)."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache, save_query_pair

    p = "atlas/x/coronal/2.00mm.jpg"
    atlas_emb = torch.full((2, 4), 1.0)
    query_emb = torch.full((2, 4), 9.0)
    save_query_pair(
        tmp_path, plane="coronal", dataset="ad_bxd",
        embeddings={p: query_emb},
    )
    chain = _CacheChain(
        atlas=_StubCache({p: atlas_emb}),
        query=QueryEmbeddingCache(tmp_path),
    )
    out = chain.lookup_by_path(p)
    assert out is not None
    assert (out == 1.0).all().item()  # atlas, not query


# ---------------------------------------------------------------------------
# CLI: --query-embedding-cache
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[grpo]\n", encoding="utf-8")
    return cfg_path


def test_train_grpo_query_cache_cli_parses(tmp_path: Path) -> None:
    """``--query-embedding-cache PATH`` parses cleanly and exposes args.query_embedding_cache."""
    cfg = _make_cfg(tmp_path)
    qcache = tmp_path / "query_cache"
    qcache.mkdir()
    args = tg._parse_args([
        "--config", str(cfg),
        "--sft-model", str(tmp_path),
        "--output-dir", str(tmp_path),
        "--query-embedding-cache", str(qcache),
    ])
    assert args.query_embedding_cache == qcache


def test_train_grpo_query_cache_cli_default_none(tmp_path: Path) -> None:
    """Default when flag omitted is None (cache disabled, mirrors atlas flag)."""
    cfg = _make_cfg(tmp_path)
    args = tg._parse_args([
        "--config", str(cfg),
        "--sft-model", str(tmp_path),
        "--output-dir", str(tmp_path),
    ])
    assert args.query_embedding_cache is None


def test_train_grpo_query_cache_cli_help_advertises_flag(tmp_path: Path) -> None:
    """``--help`` text mentions the new flag so operators can discover it."""
    import io
    import sys as _sys

    buf = io.StringIO()
    old = _sys.stdout
    _sys.stdout = buf
    try:
        with pytest.raises(SystemExit):
            tg._parse_args(["--help"])
    finally:
        _sys.stdout = old
    out = buf.getvalue()
    assert "--query-embedding-cache" in out


def test_train_grpo_query_cache_missing_dir_error_is_clear(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """When the flag points at a nonexistent dir AND the trainer would actually
    use it, surface a clear ``FileNotFoundError`` instead of silent fall-through.

    We exercise the helper that loads the cache rather than the full main()
    (which requires TRL/Unsloth)."""
    from langslice_training.embeddings.query_cache import QueryEmbeddingCache

    bogus = tmp_path / "does_not_exist"
    cache = QueryEmbeddingCache(bogus)
    # Empty cache is valid — every lookup returns None.
    assert cache.pairs() == []
    assert cache.lookup_by_path("anything.jpg") is None

    # The trainer's own validator (mirrors atlas-cache behaviour at lines
    # 712-718 of train_grpo) raises when the dir contains zero files.
    with pytest.raises(FileNotFoundError, match="query"):
        tg._validate_query_cache_loaded(cache, bogus)
