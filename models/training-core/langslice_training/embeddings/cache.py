"""Runtime lookup of precomputed SigLIP embeddings keyed by on-disk JPG path.

Cache layout on disk
====================

One ``.pt`` file per ``(atlas, plane)`` pair, written by
:mod:`models.langslice-gemma-4.training.embeddings.precompute`::

    <cache_dir>/<atlas>_<plane>.pt

Each file contains a single torch-saved dict with the structure::

    {
        "atlas": "allen_mouse_25um",
        "plane": "coronal",
        "embeddings": {basename: torch.Tensor},  # SigLIP last_hidden_state
    }

``basename`` is the on-disk JPG filename WITHOUT directory components — e.g.
``"0.62mm.jpg"``. The cache entry is the encoded SigLIP output for that exact
file. Path-based lookup avoids the quantization-collision bug that the older
``qkey``-keyed scheme had: when the corpus contains multiple non-grid-aligned
positions in the same 0.05 mm bucket (e.g. 0.62, 0.65, 0.68), each gets its
own cache entry rather than fighting for the same key.

The lookup path accepts the canonical training image path
``atlas/<atlas>/<plane>/<basename>`` so the SFT collator can match
``tool_result.image_paths`` entries directly without parsing them itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from langslice_training.rl.common.atlas_grid import canonicalize_atlas_name

Plane = Literal["coronal", "sagittal", "horizontal"]

# Canonical training-image-path shape:
#   atlas/<atlas>/<plane>/<basename>
# Tolerates leading "./" or absolute prefixes — we only care about the last 4
# path segments. ``<basename>`` is the leaf JPG/PNG filename and is preserved
# as-is so it can be used directly as the per-shard cache key.
_ATLAS_PATH_RE = re.compile(
    r"(?:^|[\\/])atlas[\\/]"
    r"(?P<atlas>[^\\/]+)[\\/]"
    r"(?P<plane>coronal|sagittal|horizontal)[\\/]"
    r"(?P<basename>[^\\/]+\.(?:png|jpg))$"
)

# When precomputing on a brand-new corpus we need a way to spell the cache
# filename — match by stripping the dataset-relative prefix.
_FILENAME_RE = re.compile(r"^(?P<atlas>.+)_(?P<plane>coronal|sagittal|horizontal)\.pt$")


@dataclass(frozen=True)
class AtlasPathParts:
    atlas: str
    plane: Plane
    basename: str


def parse_atlas_path(path: str) -> AtlasPathParts | None:
    """Return ``(atlas, plane, basename)`` for a canonical atlas image path.

    Returns ``None`` for paths that don't match the
    ``atlas/<atlas>/<plane>/<basename>`` shape. Non-atlas paths
    (queries, augmentations, etc.) hit this branch.
    """
    m = _ATLAS_PATH_RE.search(str(path).replace("\\", "/"))
    if m is None:
        return None
    plane = m.group("plane")
    return AtlasPathParts(
        atlas=m.group("atlas"),
        plane=plane,  # type: ignore[arg-type]
        basename=m.group("basename"),
    )


@dataclass
class _CacheEntry:
    """Lazy-loaded per-pair cache.

    Files are mmap'd via ``torch.load(map_location="cpu", mmap=True)`` when
    available so we don't pay the deserialization cost up front for pairs
    the trainer never touches. ``embeddings`` stays None until the first
    lookup forces a load.
    """

    path: Path
    embeddings: dict[str, torch.Tensor] | None = None

    def ensure_loaded(self) -> None:
        if self.embeddings is not None:
            return
        # mmap=True only works for CPU loads of recent torch saves; fall back
        # to plain load if the runtime doesn't support it (older torch /
        # mmap-incompatible payloads). Errors here are unrecoverable — bubble
        # them so the caller knows the cache is corrupt.
        try:
            payload = torch.load(self.path, map_location="cpu", mmap=True, weights_only=False)
        except (TypeError, RuntimeError):
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise RuntimeError(
                f"atlas embedding cache file {self.path} is malformed (missing 'embeddings' key)"
            )
        self.embeddings = payload["embeddings"]


class AtlasEmbeddingCache:
    """Lookup precomputed SigLIP embeddings by ``atlas/<atlas>/<plane>/<basename>``.

    Construct once per training run and pass to :class:`LangSliceCollator`.
    The cache discovers all ``<atlas>_<plane>.pt`` files in ``cache_dir`` at
    init time but defers reading their tensors until the first lookup for
    that pair.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._entries: dict[tuple[str, Plane], _CacheEntry] = {}
        if not self.cache_dir.is_dir():
            # Empty cache is valid — every lookup returns None and the
            # collator's hit-rate counter stays at 0. We don't raise so a
            # training run can opt into the cache progressively.
            return
        for child in self.cache_dir.iterdir():
            if not child.is_file() or child.suffix != ".pt":
                continue
            m = _FILENAME_RE.match(child.name)
            if m is None:
                # Unknown file in the cache dir; ignore. Could be a temp from
                # an interrupted precompute run.
                continue
            atlas = canonicalize_atlas_name(m.group("atlas"))
            plane = m.group("plane")
            self._entries[(atlas, plane)] = _CacheEntry(path=child)  # type: ignore[arg-type]

    def pairs(self) -> list[tuple[str, Plane]]:
        return sorted(self._entries.keys())

    def has_pair(self, atlas: str, plane: Plane) -> bool:
        return (canonicalize_atlas_name(atlas), plane) in self._entries

    def lookup(
        self,
        atlas: str,
        plane: Plane,
        basename: str,
    ) -> torch.Tensor | None:
        """Return the cached embedding for ``basename`` in this (atlas, plane), else None.

        The returned tensor is a clone — the cache may be backed by mmap, so
        callers that mutate it (e.g. moving to device) need an owning copy.
        """
        key = (canonicalize_atlas_name(atlas), plane)
        entry = self._entries.get(key)
        if entry is None:
            return None
        entry.ensure_loaded()
        assert entry.embeddings is not None  # ensure_loaded postcondition
        emb = entry.embeddings.get(basename)
        if emb is None:
            return None
        return emb.clone()

    def lookup_by_path(self, path: str) -> torch.Tensor | None:
        """Parse a canonical atlas image path and dispatch to :meth:`lookup`."""
        parts = parse_atlas_path(path)
        if parts is None:
            return None
        return self.lookup(parts.atlas, parts.plane, parts.basename)
