"""Persistent cache for SigLIP embeddings of training query (slice) images.

Sibling of :mod:`langslice_training.embeddings.cache` (which caches atlas reference images).
Where the atlas cache is keyed by ``atlas/<atlas>/<plane>/<basename>``
because every training row references the same atlas grid, the query cache
is keyed by the **full repo-relative image_path** taken from the manifest
(``data/datasets/<plane>/<dataset>/<subject>/<section_id>.jpg``).
Per-section paths are unique — they don't admit a shorter canonical key.

Cache layout on disk
====================

One ``.pt`` file per ``(plane, dataset)`` pair, written by
:mod:`langslice_training.embeddings.precompute_query`::

    <cache_dir>/<plane>__<dataset>.pt

Each file contains a single torch-saved dict::

    {
        "plane": "coronal",
        "dataset": "ad_bxd",
        "embeddings": {full_image_path: torch.Tensor},
    }

Why ``(plane, dataset)`` instead of a single global file? Two reasons:

1. **Operability.** Re-precomputing one dataset shouldn't rewrite a 60 GB
   payload. Per-pair files mean ``--max-rows`` smoke tests + per-shard
   reruns are cheap.
2. **Symmetry.** The atlas cache uses one file per ``(atlas, plane)``;
   keeping the same shape makes the splice + chain-lookup wiring trivially
   parallel.

Atomic-write contract
=====================

:func:`save_query_pair` mirrors :func:`langslice_training.embeddings.precompute._save_pair`:
torch-saves to ``<path>.partial`` then ``replace``s atomically so an
interrupted precompute never leaves a half-written file behind.

Lookup precedence vs the atlas cache
====================================

The trainer wires the two caches in a chain (atlas first). In the rare
event a path collides between the two layers, atlas wins — the atlas cache
is treated as canonical because it's bit-exact-verified by
``langslice_training.embeddings._verify_cache``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Cache filename pattern: <plane>__<dataset>.pt
# Plane is restricted to the canonical three to make the regex unambiguous;
# dataset is anything not containing the double-underscore separator.
_FILENAME_RE = re.compile(r"^(?P<plane>coronal|sagittal|horizontal)__(?P<dataset>.+)\.pt$")


@dataclass
class _QueryCacheEntry:
    """Lazy-loaded per-(plane, dataset) embedding map.

    Same lazy-load pattern as :class:`langslice_training.embeddings.cache._CacheEntry`: defer
    the torch.load until the first lookup forces it. mmap when possible.
    """

    path: Path
    embeddings: dict[str, torch.Tensor] | None = None

    def ensure_loaded(self) -> None:
        if self.embeddings is not None:
            return
        try:
            payload = torch.load(self.path, map_location="cpu", mmap=True, weights_only=False)
        except (TypeError, RuntimeError):
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise RuntimeError(
                f"query embedding cache file {self.path} is malformed (missing 'embeddings' key)"
            )
        self.embeddings = payload["embeddings"]


class QueryEmbeddingCache:
    """Lookup precomputed SigLIP embeddings for query (slice) images by path.

    Construct once per training run. Discovers every
    ``<plane>__<dataset>.pt`` file in ``cache_dir`` at init time but defers
    reading their tensors until the first lookup for that pair.

    Mirror :class:`langslice_training.embeddings.cache.AtlasEmbeddingCache` API surface:

    * ``pairs()`` returns ``[(plane, dataset), ...]``.
    * ``lookup_by_path(path)`` returns the cached embedding or ``None``.

    Empty / missing cache dir is valid: every lookup returns ``None`` so a
    training run can opt into the cache progressively.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._entries: dict[tuple[str, str], _QueryCacheEntry] = {}
        # Reverse index: image_path → (plane, dataset) for O(1) routing on
        # lookup. Built lazily as cache files are loaded.
        self._path_to_pair: dict[str, tuple[str, str]] = {}
        if not self.cache_dir.is_dir():
            return
        for child in self.cache_dir.iterdir():
            if not child.is_file() or child.suffix != ".pt":
                continue
            m = _FILENAME_RE.match(child.name)
            if m is None:
                continue
            plane = m.group("plane")
            dataset = m.group("dataset")
            self._entries[(plane, dataset)] = _QueryCacheEntry(path=child)

    def pairs(self) -> list[tuple[str, str]]:
        """List of ``(plane, dataset)`` pairs present in the cache, sorted."""
        return sorted(self._entries.keys())

    def has_pair(self, plane: str, dataset: str) -> bool:
        return (plane, dataset) in self._entries

    def _ensure_pair_loaded(self, plane: str, dataset: str) -> None:
        entry = self._entries.get((plane, dataset))
        if entry is None or entry.embeddings is not None:
            return
        entry.ensure_loaded()
        # Index every key into the reverse map so future lookups for *any*
        # path in this shard are O(1).
        assert entry.embeddings is not None
        for key in entry.embeddings:
            self._path_to_pair[key] = (plane, dataset)

    def lookup_by_path(self, path: str) -> torch.Tensor | None:
        """Return the cached embedding for ``path``, else ``None``.

        The returned tensor is a clone — the cache may be backed by mmap, so
        callers that mutate it (e.g. moving to device) need an owning copy.
        """
        # Fast path: previously-resolved.
        pair = self._path_to_pair.get(path)
        if pair is not None:
            entry = self._entries.get(pair)
            if entry is None or entry.embeddings is None:
                return None
            emb = entry.embeddings.get(path)
            return None if emb is None else emb.clone()

        # Slow path: walk shards we haven't loaded yet. To keep lookups
        # bounded, prefer the shard whose ``(plane, dataset)`` is implied
        # by the path itself (the manifest layout is canonical:
        # ``data/datasets/<plane>/<dataset>/...``).
        guessed = _guess_pair_from_path(path)
        if guessed is not None:
            self._ensure_pair_loaded(*guessed)
            pair = self._path_to_pair.get(path)
            if pair is not None:
                return self.lookup_by_path(path)
            # Guessed shard didn't contain it — fall through to the broader
            # walk so we don't silently miss a path that lives in a
            # differently-named shard (rare, but possible if the manifest
            # path convention drifts).

        # Fall-back: load remaining shards on demand (rare; typically the
        # path-prefix guess hits).
        for plane, dataset in list(self._entries.keys()):
            if (plane, dataset) == guessed:
                continue
            self._ensure_pair_loaded(plane, dataset)
            pair = self._path_to_pair.get(path)
            if pair is not None:
                return self.lookup_by_path(path)
        return None


# Canonical manifest layout: ``data/datasets/<plane>/<dataset>/<rest>``. The
# parser tolerates leading ``./`` and Windows separators.
_PATH_PAIR_RE = re.compile(
    r"(?:^|[\\/])data[\\/]datasets[\\/]"
    r"(?P<plane>coronal|sagittal|horizontal)[\\/]"
    r"(?P<dataset>[^\\/]+)[\\/]"
)


def _guess_pair_from_path(path: str) -> tuple[str, str] | None:
    """Best-effort extract ``(plane, dataset)`` from a canonical manifest path.

    Returns ``None`` for paths that don't match the canonical layout (those
    fall back to the slow shard-walk lookup).
    """
    m = _PATH_PAIR_RE.search(path.replace("\\", "/"))
    if m is None:
        return None
    return (m.group("plane"), m.group("dataset"))


def save_query_pair(
    output_dir: Path,
    *,
    plane: str,
    dataset: str,
    embeddings: dict[str, torch.Tensor],
) -> Path:
    """Atomic-write one ``<plane>__<dataset>.pt`` file under ``output_dir``.

    Mirrors :func:`langslice_training.embeddings.precompute._save_pair`: torch-save to a
    ``.partial`` and ``replace`` so an interrupted precompute leaves no
    half-written file behind.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{plane}__{dataset}.pt"
    payload: dict[str, Any] = {
        "plane": plane,
        "dataset": dataset,
        "embeddings": embeddings,
    }
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(out_path)
    return out_path
