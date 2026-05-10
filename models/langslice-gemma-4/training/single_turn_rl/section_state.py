"""Index-driven training-row builder for single-turn GRPO (Lane B).

Sibling to :mod:`single_turn_rl.terminal_states`. Where
:class:`TerminalState` records per-trace fetched atlas evidence (Lane A,
SFT-corpus walker), :class:`SectionState` is built directly from a
manifest :class:`Section` (provided by Task 2's :class:`ManifestIndex`)
plus a deterministic, plane-spanning *atlas slate* (Lane B). The slate
is the same evenly-spaced bracket of atlas references the production
Gemini 3 Pro pipeline uses; the model brackets the answer between two
slate positions and interpolates to submit at finer precision.

Why a separate row builder
--------------------------

The Lane A walker only ships rows that the SFT teacher already solved
(strict/rescued tier). That means the policy never sees the long tail of
hard manifest sections. Lane B unlocks that pool: any manifest row whose
images exist on disk becomes a training row. The trade-off is the
agent-collected atlas evidence is gone — replaced with the canonical
slate, which carries no per-target zoom hint.

Atlas slate persistence
-----------------------

For each ``(atlas, plane)`` pair the slate is computed once and cached as
``<root>/atlas/<atlas>/<plane>/_slate_<n>.json`` so a fresh trainer
process doesn't repeat the work. The JSON shape is stable and tested
(see :func:`save_canonical_slate`); image files are NOT regenerated —
the slate just references the existing pre-rendered atlas grid at
``data/atlas/<atlas>/<plane>/<X.XX>mm.jpg``.

Scope discipline
----------------

This module is purely additive over Lane A. It never modifies
:mod:`terminal_states`, :mod:`manifest_index`, or :mod:`dataset` —
Task 6 will wire the integration. Image files are never written; only
slate-metadata JSON is created.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rlvr.atlas_grid import GRID_STEP_MM as _GRID_STEP_MM

# Lane A sibling — reuse the cached BrainGlobe range helper so the two
# lanes don't pay the multi-second atlas-load cost twice in one process.
from .manifest_index import ManifestIndex, Section
from .terminal_states import _atlas_valid_range_mm

Plane = Literal["coronal", "sagittal", "horizontal"]


# ---------------------------------------------------------------------------
# CanonicalSlate dataclass + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalSlate:
    """One canonical, plane-spanning atlas-reference slate.

    ``positions_mm`` and ``image_paths`` are parallel — index ``i`` is the
    same slate slot in both. Paths are repo-relative strings of the form
    ``data/atlas/<atlas>/<plane>/<X.XX>mm.jpg`` so the JSONL stays
    portable across machines that share the same checkout. The slate is
    derived deterministically from the atlas's valid range; two builds
    of the same ``(atlas, plane, n_positions)`` always yield the same
    slate, which keeps the trainer's curriculum tick reproducible.
    """

    atlas_name: str
    plane: str
    positions_mm: tuple[float, ...]
    image_paths: tuple[str, ...]


def _snap_to_grid(position_mm: float) -> float:
    """Snap ``position_mm`` to the nearest 0.05 mm grid point.

    ``round(x / step) * step`` introduces float jitter (e.g. 4.55 ends up
    at 4.550000000000001), which cascades into mismatched filenames
    downstream. Round to two decimal places to keep the snapped value
    representable exactly by ``f"{x:.2f}"``.
    """
    snapped = round(position_mm / _GRID_STEP_MM) * _GRID_STEP_MM
    return round(snapped, 2)


def _atlas_image_path(atlas_name: str, plane: str, position_mm: float) -> str:
    """Build the canonical repo-relative atlas image path for a slate slot."""
    return f"data/atlas/{atlas_name}/{plane}/{position_mm:.2f}mm.jpg"


def _evenly_spaced_positions(
    pos_lo: float, pos_hi: float, n_positions: int
) -> list[float]:
    """Compute ``n_positions`` evenly-spaced positions across ``[pos_lo, pos_hi]``.

    For ``n_positions=1`` the single position is the midpoint (anything
    else would arbitrarily favour an endpoint). For ``n_positions >= 2``
    the first and last positions are the bracketing extremes — this
    matches the production Gemini 3 Pro slate where the model needs to
    see the anatomical poles before bracketing.
    """
    if n_positions <= 0:
        raise ValueError(f"n_positions must be positive, got {n_positions}")
    if pos_hi < pos_lo:
        raise ValueError(
            f"pos_hi ({pos_hi}) must be >= pos_lo ({pos_lo}) for slate generation"
        )
    if n_positions == 1:
        return [(pos_lo + pos_hi) / 2.0]
    step = (pos_hi - pos_lo) / (n_positions - 1)
    return [pos_lo + i * step for i in range(n_positions)]


def build_canonical_slate(
    atlas_name: str,
    plane: str,
    *,
    n_positions: int = 9,
) -> CanonicalSlate:
    """Compute the canonical slate for ``(atlas_name, plane)``.

    Walks ``n_positions`` evenly-spaced positions across the atlas's
    valid range, snaps each to the 0.05 mm grid (so the resulting
    filenames match the on-disk pre-rendered atlas images), and returns
    a :class:`CanonicalSlate`. Snapped positions are deduplicated in
    order — at very small valid ranges (or very high ``n_positions``)
    multiple slots could collapse onto the same grid point, and silently
    keeping duplicates would mean the trainer pays N>1 image-decode
    costs for a single slice's worth of evidence.

    The valid range comes from the cached
    :func:`single_turn_rl.terminal_states._atlas_valid_range_mm` helper.
    """
    pos_lo, pos_hi = _atlas_valid_range_mm(atlas_name, plane)  # type: ignore[arg-type]
    raw_positions = _evenly_spaced_positions(pos_lo, pos_hi, n_positions)

    # Snap each position to the grid, clamping to stay strictly inside
    # the atlas's valid range. Without the inward clamp at the high end
    # (analogous to ``rlvr.atlas_grid.build_atlas_grid``) a snapped
    # position could round past ``pos_hi`` and reference an image file
    # that was never rendered.
    snapped: list[float] = []
    seen: set[float] = set()
    for raw in raw_positions:
        clipped = max(pos_lo, min(pos_hi, raw))
        snap = _snap_to_grid(clipped)
        if snap > pos_hi:
            snap = round(snap - _GRID_STEP_MM, 2)
        if snap < pos_lo:
            snap = round(snap + _GRID_STEP_MM, 2)
        if snap in seen:
            continue
        seen.add(snap)
        snapped.append(snap)

    image_paths = tuple(
        _atlas_image_path(atlas_name, plane, p) for p in snapped
    )
    return CanonicalSlate(
        atlas_name=atlas_name,
        plane=plane,
        positions_mm=tuple(snapped),
        image_paths=image_paths,
    )


def _slate_json_path(root: Path, atlas_name: str, plane: str, n_positions: int) -> Path:
    """Resolve the on-disk JSON path for a ``(atlas, plane, n)`` slate."""
    return root / "atlas" / atlas_name / plane / f"_slate_{n_positions}.json"


def save_canonical_slate(slate: CanonicalSlate, root: Path) -> Path:
    """Persist slate metadata to ``<root>/atlas/<atlas>/<plane>/_slate_<n>.json``.

    Creates parent directories on demand. Returns the JSON path. The
    written file is the canonical handoff between trainer startup and
    cached slate resolution; the schema is fixed (and tested):

    * ``atlas_name`` / ``plane`` / ``n_positions``: identifying the slate
    * ``positions_mm`` / ``image_paths``: parallel slate slots
    * ``valid_range_mm``: ``[pos_lo, pos_hi]`` of the atlas/plane (handy
      for downstream checks; computing it requires loading the atlas).
    """
    n_positions = len(slate.positions_mm)
    target = _slate_json_path(root, slate.atlas_name, slate.plane, n_positions)
    target.parent.mkdir(parents=True, exist_ok=True)
    pos_lo, pos_hi = _atlas_valid_range_mm(slate.atlas_name, slate.plane)  # type: ignore[arg-type]
    payload: dict[str, Any] = {
        "atlas_name": slate.atlas_name,
        "plane": slate.plane,
        "n_positions": n_positions,
        "positions_mm": list(slate.positions_mm),
        "image_paths": list(slate.image_paths),
        "valid_range_mm": [float(pos_lo), float(pos_hi)],
    }
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return target


def load_canonical_slate(
    root: Path, atlas_name: str, plane: str, *, n_positions: int = 9
) -> CanonicalSlate | None:
    """Round-trip a slate from the cached JSON. Returns ``None`` if absent.

    Only ``atlas_name``, ``plane``, ``positions_mm``, and ``image_paths``
    are read back into the dataclass — the on-disk ``valid_range_mm`` is
    diagnostic only and re-derived lazily by callers that need it.
    """
    path = _slate_json_path(root, atlas_name, plane, n_positions)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return CanonicalSlate(
        atlas_name=str(payload["atlas_name"]),
        plane=str(payload["plane"]),
        positions_mm=tuple(float(p) for p in payload.get("positions_mm", [])),
        image_paths=tuple(str(p) for p in payload.get("image_paths", [])),
    )


def get_or_build_canonical_slate(
    atlas_name: str,
    plane: str,
    *,
    root: Path,
    n_positions: int = 9,
) -> CanonicalSlate:
    """Cache-aware slate fetch: load if present, otherwise build + save + return.

    ``root`` is the slate-cache root (typically the repo's
    ``models/langslice-gemma-4/data/`` so the JSON sits next to the
    pre-rendered atlas images). The cache is keyed by ``n_positions`` so
    a smoke run with ``n=5`` and a real run with ``n=9`` don't clobber
    each other.
    """
    cached = load_canonical_slate(root, atlas_name, plane, n_positions=n_positions)
    if cached is not None:
        return cached
    slate = build_canonical_slate(
        atlas_name, plane, n_positions=n_positions,
    )
    save_canonical_slate(slate, root)
    return slate


# ---------------------------------------------------------------------------
# SectionState dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionState:
    """Index-driven analog of :class:`TerminalState`.

    Mirrors :class:`TerminalState`'s public shape (so :mod:`dataset` can
    consume either source via a thin adapter) with two differences:

    * ``dataset`` is recorded explicitly — the manifest's composite key is
      ``(plane, dataset, section_id)``, and the trainer needs to round-trip
      that key for live-difficulty updates.
    * ``atlas_slate_paths`` / ``atlas_slate_positions_mm`` replace
      :class:`TerminalState`'s per-trace ``atlas_image_paths`` /
      ``fetched_positions_mm`` — the slate is deterministic, not
      teacher-derived.

    ``difficulty_score`` is the live-rollout difficulty (or seeded
    slicebench score) at row build time. It's ``None`` for cold-start
    sections that haven't been observed yet — the curriculum sampler
    treats ``None`` as legitimate "give it a chance" material rather
    than dropping it.
    """

    section_id: str
    subject_id: str
    atlas_name: str
    plane: str
    dataset: str
    valid_range_mm: tuple[float, float]
    ground_truth_mm: float
    query_image_path: str
    atlas_slate_paths: tuple[str, ...]
    atlas_slate_positions_mm: tuple[float, ...]
    difficulty_score: float | None

    @classmethod
    def from_section(
        cls,
        section: Section,
        *,
        slate: CanonicalSlate,
        difficulty_score: float | None,
    ) -> SectionState:
        """Build a :class:`SectionState` from a :class:`Section` + slate.

        The valid range is resolved via the cached
        :func:`_atlas_valid_range_mm` helper so callers don't have to
        load the BrainGlobe atlas themselves. ``difficulty_score`` is
        passed through verbatim — :func:`iter_section_states` looks it
        up via :meth:`ManifestIndex.live_difficulty`.
        """
        pos_lo, pos_hi = _atlas_valid_range_mm(section.atlas, section.plane)  # type: ignore[arg-type]
        return cls(
            section_id=section.section_id,
            subject_id=section.subject_id,
            atlas_name=section.atlas,
            plane=section.plane,
            dataset=section.dataset,
            valid_range_mm=(float(pos_lo), float(pos_hi)),
            ground_truth_mm=float(section.position_mm),
            query_image_path=section.image_path,
            atlas_slate_paths=tuple(slate.image_paths),
            atlas_slate_positions_mm=tuple(slate.positions_mm),
            difficulty_score=(
                None if difficulty_score is None else float(difficulty_score)
            ),
        )


# ---------------------------------------------------------------------------
# iter_section_states generator
# ---------------------------------------------------------------------------


def iter_section_states(
    *,
    index: ManifestIndex,
    plane: str,
    atlas: str,
    split: str | None = None,
    slate_root: Path,
    n_positions: int = 9,
    require_query_on_disk: bool = True,
    require_atlas_on_disk: bool = True,
    repo_root: Path | None = None,
) -> Iterator[SectionState]:
    """Yield one :class:`SectionState` per matching :class:`Section`.

    Filters
    -------
    Sections are selected via :meth:`ManifestIndex.query` on
    ``(plane, atlas, split)``. ``split=None`` keeps every split (useful
    for smoke runs that don't care about the eval/rlvr/sft partitioning).

    Slate handling
    --------------
    Exactly one slate is built (or loaded from cache) per ``(atlas,
    plane)`` pair — every yielded :class:`SectionState` for that pair
    references the same slate object.

    Disk checks
    -----------
    When ``require_query_on_disk=True`` (default), sections whose query
    image is missing under ``repo_root`` are silently skipped. Same for
    ``require_atlas_on_disk`` — but if any single slate image is missing
    the entire pair is dropped (a slate is all-or-nothing — partial
    slates are not a supported training shape).

    ``repo_root`` defaults to ``Path.cwd()``. The image paths in the
    section + slate are repo-relative, so the on-disk check resolves
    them under ``repo_root``.
    """
    base = (repo_root or Path.cwd()).resolve()
    sections = index.query(plane=plane, atlas=atlas, split=split)
    if not sections:
        return

    slate = get_or_build_canonical_slate(
        atlas, plane, root=slate_root, n_positions=n_positions,
    )

    if require_atlas_on_disk:
        missing = [p for p in slate.image_paths if not (base / p).is_file()]
        if missing:
            # Whole slate is unusable — drop every section for this
            # (atlas, plane) rather than ship a partial slate. Surface
            # the reason via stderr so an operator can re-render the
            # missing files instead of guessing.
            print(
                f"[iter_section_states] atlas slate incomplete for "
                f"({atlas!r}, {plane!r}); missing {len(missing)} image(s) "
                f"under {base}; first: {missing[0]!r}",
                file=sys.stderr,
                flush=True,
            )
            return

    for section in sections:
        if require_query_on_disk and not (base / section.image_path).is_file():
            continue
        live = index.live_difficulty(section.plane, section.dataset, section.section_id)
        difficulty = None if live is None else float(live.score)
        yield SectionState.from_section(
            section, slate=slate, difficulty_score=difficulty,
        )


__all__ = [
    "CanonicalSlate",
    "SectionState",
    "build_canonical_slate",
    "get_or_build_canonical_slate",
    "iter_section_states",
    "load_canonical_slate",
    "save_canonical_slate",
]
