"""TraceIterator: streaming corpus -> RenderedExample dispatch with augmentations.

Wraps :func:`iter_canonical_traces` (or any caller-supplied iterable of
:class:`CanonicalTrace`) with a seeded augmentation pipeline and dispatches
to one of the four rendering modes. Manifest ground-truth lookup uses the
real :class:`TraceManifestRow` shape on disk — keyed on
``(plane, subject_id, query_image_stem)`` where subject_id is taken from
``metadata['subject_id']`` (falling back to the manifest row's ``id``) and
each (image, truth_position_mm) pair is unrolled into its own index entry.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from .augmentations import AtlasFetchJitter
from .manifest import TraceManifestRow, load_manifest
from .parser import iter_canonical_traces
from .renderers import (
    render_isft_prefix,
    render_rl_prefix,
    render_sft_answer_only,
    render_sft_full,
)
from .renderers._common import AtlasMetaCache
from .schema import CanonicalTrace, RenderedExample

RenderMode = Literal["sft_full", "sft_answer_only", "rl_prefix", "isft_prefix"]

# Augmentation keys recognized in the ``augmentations`` config dict, mapped to
# their required sub-fields. Required-field validation runs at ``__init__``
# so misconfiguration fails loudly at construction rather than via a bare
# ``KeyError`` deep inside ``_build_augmentations`` on the first yielded item.
_KNOWN_AUGMENTATIONS: dict[str, tuple[str, ...]] = {
    "atlas_fetch_jitter": ("sigma_mm", "max_calls_jittered"),
}


class TraceIterator:
    """Stream CanonicalTrace instances from a corpus, apply augmentations, render in mode.

    Parameters
    ----------
    corpus
        Path to an SFT-corpus JSONL OR an iterable of pre-parsed
        :class:`CanonicalTrace` instances. When a path is supplied,
        :func:`iter_canonical_traces` handles streaming + dataset_root
        attachment. **Non-Path iterables are materialized eagerly at
        construction** (consuming memory proportional to corpus size) so
        that ``iter(my_iterator)`` is safely re-callable — supplying a
        one-shot generator would otherwise yield nothing on the second
        pass.
    mode
        One of ``"sft_full"``, ``"sft_answer_only"``, ``"rl_prefix"``,
        ``"isft_prefix"``.
    seed
        Seeds a fresh :class:`random.Random` instantiated per ``__iter__``
        call. The same seed + corpus + augmentation config yields an
        identical output sequence on every iteration of the same iterator
        instance.
    augmentations
        Optional config dict mapping augmentation name -> kwargs, e.g.::

            {"atlas_fetch_jitter": {"sigma_mm": 0.05, "max_calls_jittered": 2}}

        Only ``"atlas_fetch_jitter"`` is recognized in this pass. Unknown
        keys raise :class:`NotImplementedError` at construction (planned
        keys like ``"clahe_mix_toggle"`` are deferred to Pass 4). The
        augmentation instances are rebuilt at the start of each
        ``__iter__`` call using a fresh :class:`random.Random` seeded with
        ``seed`` so re-iteration is deterministic.
    manifest
        REQUIRED for ``"rl_prefix"`` mode AND for ``"sft_answer_only"``
        with ``prefer_ground_truth=True`` — both need a ground_truth_mm
        lookup. Expects the standard :class:`TraceManifestRow` JSONL
        shape (id/kind/images[]/atlas/plane/truth_positions_mm[]). Each
        (image, position) pair is unrolled into an index entry keyed on
        ``(plane, subject_id, query_image_stem)``, where ``subject_id``
        comes from ``metadata['subject_id']`` (falling back to the
        manifest row's ``id``).
    atlas_meta_cache
        REQUIRED for all modes — every renderer builds a system prompt
        from atlas metadata. Reuse a single instance per training run.
    prefer_ground_truth
        Forwarded to :func:`render_sft_answer_only`. When True the
        iterator looks up the GT from the manifest per trace and passes
        it through; ``manifest`` must also be supplied (enforced at
        construction).
    grid_resolver
        Optional ``(atlas_name, plane) -> list[float]`` used by
        :class:`AtlasFetchJitter` to snap jittered positions to the
        path-keyed embedding cache grid. When omitted, jittered values
        are used as-is (acceptable for tests, NOT recommended for
        training — see :class:`AtlasFetchJitter`).
    """

    def __init__(
        self,
        corpus: Path | Iterable[CanonicalTrace],
        mode: RenderMode,
        *,
        seed: int = 42,
        augmentations: dict[str, Any] | None = None,
        manifest: Path | None = None,
        atlas_meta_cache: AtlasMetaCache | None = None,
        prefer_ground_truth: bool = False,
        grid_resolver: Callable[[str, str], list[float]] | None = None,
    ):
        if mode not in ("sft_full", "sft_answer_only", "rl_prefix", "isft_prefix"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "rl_prefix" and manifest is None:
            raise ValueError(
                "rl_prefix mode requires a manifest path for ground_truth lookup"
            )
        if prefer_ground_truth and manifest is None:
            raise ValueError(
                "prefer_ground_truth=True requires a manifest path to look up "
                "ground_truth_mm; pass manifest=..."
            )
        if atlas_meta_cache is None:
            raise ValueError("atlas_meta_cache is required")
        # Validate the augmentations config eagerly so misconfiguration
        # fails at construction, not mid-iteration.
        aug_config: dict[str, Any] = dict(augmentations or {})
        for key, kwargs in aug_config.items():
            if key not in _KNOWN_AUGMENTATIONS:
                raise NotImplementedError(
                    f"Augmentation {key!r} is not yet implemented "
                    "(planned for Pass 4)"
                )
            if not isinstance(kwargs, dict):
                raise ValueError(
                    f"augmentations[{key!r}] must be a dict of sub-field "
                    f"kwargs; got {type(kwargs).__name__}"
                )
            for required in _KNOWN_AUGMENTATIONS[key]:
                if required not in kwargs:
                    raise ValueError(
                        f"augmentations[{key!r}] missing required field "
                        f"{required!r}"
                    )

        self.corpus = corpus
        self.mode = mode
        self.seed = int(seed)
        self._aug_config = aug_config
        self.atlas_meta_cache = atlas_meta_cache
        self.prefer_ground_truth = prefer_ground_truth
        self._grid_resolver = grid_resolver
        # Materialize non-Path corpora at construction so the iterator
        # is safely re-iterable (a one-shot generator would yield nothing
        # on the second pass). Path inputs are streamed lazily on each
        # __iter__ call via iter_canonical_traces.
        self._corpus_materialized: list[CanonicalTrace] | None
        if isinstance(corpus, Path):
            self._corpus_materialized = None
        else:
            self._corpus_materialized = list(corpus)
        self._gt_index: dict[tuple[str, str, str], float] | None = None
        if manifest is not None:
            self._gt_index = self._build_gt_index(manifest)

    @staticmethod
    def _build_gt_index(manifest: Path) -> dict[tuple[str, str, str], float]:
        """Load a trace-collection manifest JSONL and build a GT lookup index.

        Adapted to the real :class:`TraceManifestRow` shape on disk
        (``id``/``kind``/``images``/``atlas``/``plane``/``truth_positions_mm``),
        which the spec author's described shape does not match exactly.
        Specifically:

        - ``subject_id`` is not a top-level manifest field. We pull it
          from ``metadata['subject_id']`` if present, falling back to the
          manifest row's ``id``. The corpus emitter at
          ``_local/trace_collection/build_sft_corpus.py:654`` writes
          ``metadata['subject_id']`` for the same reason.
        - ``query_image_path`` is not a top-level field either. Each row
          carries a parallel ``images``/``truth_positions_mm`` pair, so we
          unroll every (image, position) into its own index entry and
          key on ``Path(image).stem`` to match
          :meth:`_lookup_gt`'s key construction.

        Rows that fail :func:`parse_manifest_record` (malformed) are
        propagated up — the iterator's failure mode for a broken manifest
        should be loud, not silent.
        """
        index: dict[tuple[str, str, str], float] = {}
        rows: list[TraceManifestRow] = load_manifest(manifest)
        for row in rows:
            subject_id = str(row.metadata.get("subject_id") or row.id)
            if len(row.images) != len(row.truth_positions_mm):
                # parse_manifest_record already enforces this for grouped
                # records; keep the guard for defensive iteration.
                continue
            for image, pos in zip(row.images, row.truth_positions_mm, strict=True):
                key = (row.plane, subject_id, Path(image).stem)
                index[key] = float(pos)
        return index

    def _build_augmentations(self) -> list[Any]:
        """Instantiate augmentation callables from the stored config dict.

        Called at the start of every ``__iter__`` invocation so that
        re-iteration with the same seed is deterministic. Augmentations
        consume random state at call time via the rng threaded through
        ``__iter__`` rather than at construction, so this helper takes no
        rng of its own.
        """
        augs: list[Any] = []
        for key, kwargs in self._aug_config.items():
            if key == "atlas_fetch_jitter":
                augs.append(
                    AtlasFetchJitter(
                        sigma_mm=float(kwargs["sigma_mm"]),
                        max_calls_jittered=int(kwargs["max_calls_jittered"]),
                        grid_resolver=self._grid_resolver,
                    )
                )
            else:
                # Defensive: __init__ already rejected unknown keys, but
                # if a subclass ever bypasses that guard, fail loudly here.
                raise NotImplementedError(
                    f"Augmentation {key!r} is not yet implemented "
                    "(planned for Pass 4)"
                )
        return augs

    def _source_iter(self) -> Iterator[CanonicalTrace]:
        if isinstance(self.corpus, Path):
            yield from iter_canonical_traces(self.corpus)
        else:
            assert self._corpus_materialized is not None
            yield from self._corpus_materialized

    def __iter__(self) -> Iterator[RenderedExample]:
        rng = random.Random(self.seed)
        augs = self._build_augmentations()
        for canonical in self._source_iter():
            for aug in augs:
                canonical = aug(canonical, rng)
            yield self._dispatch(canonical)

    def _dispatch(self, canonical: CanonicalTrace) -> RenderedExample:
        if self.mode == "sft_full":
            return render_sft_full(canonical, atlas_meta_cache=self.atlas_meta_cache)
        if self.mode == "sft_answer_only":
            # For answer-only with prefer_ground_truth, look up GT from manifest.
            if self.prefer_ground_truth and self._gt_index is not None:
                gt = self._lookup_gt(canonical)
                return render_sft_answer_only(
                    canonical,
                    atlas_meta_cache=self.atlas_meta_cache,
                    prefer_ground_truth=True,
                    ground_truth_mm=gt,
                )
            return render_sft_answer_only(
                canonical,
                atlas_meta_cache=self.atlas_meta_cache,
                prefer_ground_truth=self.prefer_ground_truth,
                ground_truth_mm=None,
            )
        if self.mode == "rl_prefix":
            gt = self._lookup_gt(canonical)
            return render_rl_prefix(
                canonical,
                atlas_meta_cache=self.atlas_meta_cache,
                ground_truth_mm=gt,
            )
        if self.mode == "isft_prefix":
            return render_isft_prefix(canonical, atlas_meta_cache=self.atlas_meta_cache)
        raise AssertionError(f"unreachable mode: {self.mode!r}")

    def _lookup_gt(self, canonical: CanonicalTrace) -> float:
        assert self._gt_index is not None
        if not canonical.query_image_paths:
            raise ValueError(
                f"trace {canonical.subject_id!r} has no query_image_paths"
            )
        stem = Path(canonical.query_image_paths[0]).stem
        key = (canonical.plane, canonical.subject_id, stem)
        if key not in self._gt_index:
            raise KeyError(
                f"no manifest GT for (plane={canonical.plane!r}, "
                f"subject={canonical.subject_id!r}, image_stem={stem!r})"
            )
        return self._gt_index[key]
