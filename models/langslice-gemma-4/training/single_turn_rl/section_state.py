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
``models/langslice-gemma-4/data/atlas/<atlas>/<plane>/<X.XX>mm.jpg``
(canonical per :data:`langslice_traces.CANONICAL_ATLAS_ROOT`).

Scope discipline
----------------

This module is purely additive over Lane A. It never modifies
:mod:`terminal_states`, :mod:`manifest_index`, or :mod:`dataset` —
Task 6 will wire the integration. Image files are never written; only
slate-metadata JSON is created.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rlvr.atlas_grid import GRID_STEP_MM as _GRID_STEP_MM

# Lane A sibling — reuse the cached BrainGlobe range helper so the two
# lanes don't pay the multi-second atlas-load cost twice in one process.
# ``_validate_embedding_cache`` is imported (not factored into a shared
# helper) because Task 3's terminal_states.py is closed; importing keeps
# the helper a single source of truth without modifying that module.
from .manifest_index import ManifestIndex, Section
from .terminal_states import _atlas_valid_range_mm, _validate_embedding_cache

logger = logging.getLogger(__name__)

Plane = Literal["coronal", "sagittal", "horizontal"]

# Tag stamped on rows synthesized by langslice_traces.generate_trace under
# the Lane B strategy. Downstream consumers (curriculum, eval splits) use
# this exact string to separate generator-fabricated rows from the
# deterministic canonical slate.
RANDOMIZED_LANE_B_SOURCE: str = "procedural_generator:lane_b"

# Tag stamped on rows synthesized by langslice_traces.generate_trace under
# the Lane A (multi-step prefix) strategy. The Lane A flat slate carries the
# positions/paths of every fetch the procedural generator emitted across all
# tool steps; downstream consumers (curriculum, eval splits) use this exact
# string to distinguish Lane A synthetic prefixes from Lane B slates and from
# the deterministic canonical slate.
RANDOMIZED_LANE_A_SOURCE: str = "procedural_generator:lane_a"


# ---------------------------------------------------------------------------
# CanonicalSlate dataclass + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalSlate:
    """One canonical, plane-spanning atlas-reference slate.

    ``positions_mm`` and ``image_paths`` are parallel — index ``i`` is the
    same slate slot in both. Paths are repo-relative strings of the form
    ``models/langslice-gemma-4/data/atlas/<atlas>/<plane>/<X.XX>mm.jpg``
    (canonical per :data:`langslice_traces.CANONICAL_ATLAS_ROOT`) so the
    JSONL stays portable across machines that share the same checkout and
    matches the prefix the trainer expects when resolving images via
    ``repo_root / p``. The slate is derived deterministically from the
    atlas's valid range; two builds of the same
    ``(atlas, plane, n_positions)`` always yield the same slate, which
    keeps the trainer's curriculum tick reproducible.
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
    """Build the canonical repo-relative atlas image path for a slate slot.

    The path lands under :data:`langslice_traces.CANONICAL_ATLAS_ROOT` so
    the trainer's ``repo_root / p`` resolver finds the on-disk tile
    rendered by the atlas embedding cache builder. Lane A teacher
    rows already use this prefix; we route Lane B through the shared
    :func:`langslice_traces.canonical_atlas_repo_path` helper so a single
    convention covers both lanes.
    """
    # Local import keeps this module importable in environments that have
    # not installed the langslice_traces dependency cone.
    from langslice_traces import canonical_atlas_repo_path  # noqa: PLC0415

    return canonical_atlas_repo_path(
        f"atlas/{atlas_name}/{plane}/{position_mm:.2f}mm.jpg"
    )


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

    Image paths are normalized through
    :func:`langslice_traces.canonical_atlas_repo_path` so older cached
    JSON files that used the legacy ``data/atlas/...`` prefix come back
    in the canonical ``models/langslice-gemma-4/data/atlas/...`` form
    expected by the trainer's ``repo_root / p`` resolver. The helper is
    idempotent on already-canonical paths.
    """
    path = _slate_json_path(root, atlas_name, plane, n_positions)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Local import: keeps this module importable when langslice_traces is
    # not on the path (e.g. running unit-only test selections that don't
    # exercise the synthetic path).
    from langslice_traces import canonical_atlas_repo_path  # noqa: PLC0415

    return CanonicalSlate(
        atlas_name=str(payload["atlas_name"]),
        plane=str(payload["plane"]),
        positions_mm=tuple(float(p) for p in payload.get("positions_mm", [])),
        image_paths=tuple(
            canonical_atlas_repo_path(str(p))
            for p in payload.get("image_paths", [])
        ),
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

    ``source`` / ``quality`` are downstream-filtering provenance fields
    (default empty so the deterministic-slate path stays unchanged).
    The randomized-slate path stamps ``source=RANDOMIZED_LANE_B_SOURCE``
    and ``quality={"strategy": "lane_b_broad_slate",
    "gt_fraction_in_slate": <float>}``.
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
    source: str = ""
    quality: dict[str, Any] | None = None

    @classmethod
    def from_section(
        cls,
        section: Section,
        *,
        slate: CanonicalSlate,
        difficulty_score: float | None,
        source: str = "",
        quality: dict[str, Any] | None = None,
    ) -> SectionState:
        """Build a :class:`SectionState` from a :class:`Section` + slate.

        The valid range is resolved via the cached
        :func:`_atlas_valid_range_mm` helper so callers don't have to
        load the BrainGlobe atlas themselves. ``difficulty_score`` is
        passed through verbatim — :func:`iter_section_states` looks it
        up via :meth:`ManifestIndex.live_difficulty`.

        ``source`` / ``quality`` default to empty so the deterministic
        Lane B path stays a no-op stamp; the randomized path passes them
        explicitly.
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
            source=str(source),
            quality=dict(quality) if quality is not None else None,
        )


# ---------------------------------------------------------------------------
# Randomized Lane-B slate (per-row sample from the procedural generator)
# ---------------------------------------------------------------------------


def _repo_relative_atlas_path(generator_path: str) -> str:
    """Map any generator atlas path to the canonical repo-relative form.

    Thin shim over :func:`langslice_traces.canonical_atlas_repo_path` —
    Lane A (synthetic) and Lane B (randomized slate) consumers must agree
    on a single prefix so the trainer's ``repo_root / p`` resolver lands
    on the on-disk tile every time. Kept as a module-level helper so the
    name still works in this module's namespace (callers / tests may
    import it directly).
    """
    # Local import keeps this module importable in environments that have
    # not installed the langslice_traces dependency cone.
    from langslice_traces import canonical_atlas_repo_path  # noqa: PLC0415

    return canonical_atlas_repo_path(str(generator_path))


def build_randomized_section_state(
    section: Section,
    *,
    atlas_grid: list[float],
    rng: random.Random,
    difficulty_score: float | None,
    strategy: str = "lane_b_broad_slate",
) -> SectionState | None:
    """Build a :class:`SectionState` whose slate is a fresh randomized
    draw from :func:`langslice_traces.generate_trace`.

    Two strategies are supported:

    ``"lane_b_broad_slate"`` (default):
        Single broad slate; ``min(positions) <= gt <= max(positions)`` is a
        hard contract enforced by the generator. Quality stamps record the
        realized ``gt_fraction_in_slate``. Source tag is
        :data:`RANDOMIZED_LANE_B_SOURCE`.

    ``"lane_a_prefix"``:
        Multi-step Lane A prefix. ``tool_steps`` from every emitted step are
        flattened into one parallel ``(positions, paths)`` pair preserving
        order (step 0's positions first, then step 1, etc.). Lane A is a
        sweep, not a slate — there is no GT-bracket invariant; GT may sit
        anywhere relative to the flat positions list. Quality stamps record
        the actual ``n_tool_steps``. Source tag is
        :data:`RANDOMIZED_LANE_A_SOURCE`.

    Returns ``None`` if generation fails closed — empty tool steps,
    mismatched positions vs paths, or a generator-side ``ValueError`` /
    ``AssertionError``. The caller treats ``None`` as a soft skip; the
    iterator surfaces a counter for those.
    """
    # Local import — keeps section_state importable in environments
    # without torch (the langslice_traces grid loader is torch-backed
    # but generate_trace itself isn't).
    from langslice_traces import generate_trace  # noqa: PLC0415

    try:
        trace = generate_trace(
            image_path=section.image_path,
            ground_truth_mm=float(section.position_mm),
            plane=section.plane,  # type: ignore[arg-type]
            atlas_name=section.atlas,
            atlas_version="",
            subject_id=section.subject_id,
            grid=atlas_grid,
            strategy=strategy,  # type: ignore[arg-type]
            rng=rng,
        )
    except (ValueError, AssertionError) as exc:
        logger.warning(
            "randomized %s generate_trace failed for %s/%s/%s: %s",
            strategy, section.plane, section.dataset, section.section_id, exc,
        )
        return None

    if not trace.tool_steps:
        # Generator surfaced zero usable steps for this row (e.g. dedupe
        # collapsed every step to <2 positions). Soft-skip with a warning
        # so the caller can count drops without crashing the run.
        logger.warning(
            "randomized %s emitted zero tool_steps for %s/%s/%s; skipping",
            strategy, section.plane, section.dataset, section.section_id,
        )
        return None

    if strategy == "lane_b_broad_slate":
        # Lane B is single-slate by contract; multiple steps would mean
        # the generator's behavior changed and we'd be silently picking
        # the first one. Fail closed.
        if len(trace.tool_steps) != 1:
            logger.warning(
                "randomized lane_b emitted %d tool_steps (expected 1) for %s/%s/%s",
                len(trace.tool_steps), section.plane,
                section.dataset, section.section_id,
            )
            return None

    # Flatten every tool_step's (positions_mm, result_image_paths) pair into
    # one parallel (flat_positions, flat_paths). Preserves order — step 0's
    # positions first, then step 1, etc.
    flat_positions: list[float] = []
    flat_paths: list[str] = []
    for step in trace.tool_steps:
        step_positions = list(step.call_args.get("positions_mm", []))
        step_paths = list(step.result_image_paths)
        if len(step_positions) != len(step_paths):
            logger.warning(
                "randomized %s position/path length mismatch for %s/%s/%s: "
                "positions=%d paths=%d",
                strategy, section.plane, section.dataset, section.section_id,
                len(step_positions), len(step_paths),
            )
            return None
        flat_positions.extend(float(p) for p in step_positions)
        flat_paths.extend(step_paths)

    if not flat_positions:
        # All tool steps were empty after parsing; treat as a soft skip.
        logger.warning(
            "randomized %s flattened to zero positions for %s/%s/%s; skipping",
            strategy, section.plane, section.dataset, section.section_id,
        )
        return None

    repo_paths = tuple(_repo_relative_atlas_path(p) for p in flat_paths)
    positions_mm = tuple(flat_positions)
    # Build the slate-equivalent (carries the same parallel pair the
    # deterministic path uses) so :meth:`SectionState.from_section` can
    # consume both via one constructor.
    slate = CanonicalSlate(
        atlas_name=section.atlas,
        plane=section.plane,
        positions_mm=positions_mm,
        image_paths=repo_paths,
    )

    if strategy == "lane_a_prefix":
        quality: dict[str, Any] = {
            "strategy": "lane_a_prefix",
            "n_tool_steps": int(len(trace.tool_steps)),
            "synthetic": True,
        }
        source = RANDOMIZED_LANE_A_SOURCE
    elif strategy == "lane_b_broad_slate":
        quality_in = dict(trace.quality or {})
        gt_fraction = float(quality_in.get("gt_fraction_in_slate", 0.5))
        quality = {
            "strategy": "lane_b_broad_slate",
            "gt_fraction_in_slate": gt_fraction,
            "synthetic": True,
        }
        source = RANDOMIZED_LANE_B_SOURCE
    else:
        # Generator would have rejected an unknown strategy with ValueError
        # and we'd have returned None above; guard defensively here too.
        raise ValueError(f"unknown strategy: {strategy!r}")

    return SectionState.from_section(
        section,
        slate=slate,
        difficulty_score=difficulty_score,
        source=source,
        quality=quality,
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
    randomized: bool = False,
    rng: random.Random | None = None,
    atlas_embedding_cache_dir: Path | None = None,
    strategy: str = "lane_b_broad_slate",
) -> Iterator[SectionState]:
    """Yield one :class:`SectionState` per matching :class:`Section`.

    Filters
    -------
    Sections are selected via :meth:`ManifestIndex.query` on
    ``(plane, atlas, split)``. ``split=None`` keeps every split (useful
    for smoke runs that don't care about the eval/rlvr/sft partitioning).

    Slate handling
    --------------
    Default (``randomized=False``): exactly one canonical slate is built
    (or loaded from cache) per ``(atlas, plane)`` pair — every yielded
    :class:`SectionState` for that pair references the same slate. The
    ``strategy`` argument is ignored on this path.

    Randomized (``randomized=True``): every section gets a fresh slate
    from :func:`langslice_traces.generate_trace` with the requested
    ``strategy`` (default ``"lane_b_broad_slate"``; the trainer wires
    ``"lane_a_prefix"`` for the multi-step prefix data source). The atlas
    grid is loaded once per ``(atlas, plane)`` pair from
    ``atlas_embedding_cache_dir`` (a ``<atlas>_<plane>.pt`` file). The
    seeded ``rng`` is reused across sections so a given seed reproduces
    the same slates run-to-run.

    Disk checks
    -----------
    When ``require_query_on_disk=True`` (default), sections whose query
    image is missing under ``repo_root`` are silently skipped. The
    deterministic path additionally validates every slate image up front
    via ``require_atlas_on_disk``; the randomized path skips that bulk
    check because the per-section slate is different each call.

    ``repo_root`` defaults to ``Path.cwd()``. The image paths in the
    section + slate are repo-relative, so the on-disk check resolves
    them under ``repo_root``.
    """
    base = (repo_root or Path.cwd()).resolve()
    sections = index.query(plane=plane, atlas=atlas, split=split)
    if not sections:
        return

    if randomized:
        # Randomized: per-row slate via the procedural generator.
        if atlas_embedding_cache_dir is None:
            raise ValueError(
                "iter_section_states(randomized=True) requires "
                "atlas_embedding_cache_dir to be set."
            )
        if rng is None:
            # Seeded callers pass an explicit ``Random``; an unseeded
            # caller still works deterministically per-process if they
            # forgot, but we warn so the operator can fix it.
            rng = random.Random(1337)
            logger.warning(
                "iter_section_states(randomized=True) called without rng; "
                "using random.Random(1337)."
            )
        # Local import: torch is only needed on the randomized path.
        from langslice_traces import load_atlas_grid  # noqa: PLC0415

        try:
            atlas_grid = load_atlas_grid(
                atlas_embedding_cache_dir, atlas, plane,  # type: ignore[arg-type]
            )
        except FileNotFoundError as exc:
            print(
                f"[iter_section_states] randomized slate skip ({atlas!r}, "
                f"{plane!r}): {exc}",
                file=sys.stderr, flush=True,
            )
            return

        for section in sections:
            if require_query_on_disk and not (base / section.image_path).is_file():
                continue
            live = index.live_difficulty(
                section.plane, section.dataset, section.section_id,
            )
            difficulty = None if live is None else float(live.score)
            state = build_randomized_section_state(
                section,
                atlas_grid=atlas_grid,
                rng=rng,
                difficulty_score=difficulty,
                strategy=strategy,
            )
            if state is None:
                continue
            yield state
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cmd(args: argparse.Namespace) -> int:
    """Drive :func:`iter_section_states` and write per-row JSONL.

    The CLI is a thin wrapper meant to smoke-test the iterator + the
    ``--randomized-slate`` flag end-to-end. Real trainer paths consume
    :func:`iter_section_states` directly.
    """
    if args.randomized_slate:
        _validate_embedding_cache(args.atlas_embedding_cache)

    repo_root: Path = (args.repo_root or Path.cwd()).resolve()
    slate_root: Path = args.slate_root or (repo_root / "models" / "langslice-gemma-4" / "data")

    cache_label = (
        str(args.atlas_embedding_cache) if args.randomized_slate else "<canonical>"
    )
    print(
        f"[section_state] randomized={args.randomized_slate}, "
        f"seed={args.randomized_seed}, cache={cache_label}",
        file=sys.stderr, flush=True,
    )

    index = ManifestIndex.from_manifest_root(args.manifest_root, repo_root=repo_root)

    rng: random.Random | None = (
        random.Random(args.randomized_seed) if args.randomized_slate else None
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_synthesized = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for state in iter_section_states(
            index=index,
            plane=args.plane,
            atlas=args.atlas,
            split=args.split,
            slate_root=slate_root,
            n_positions=args.n_positions,
            require_query_on_disk=not args.no_check_disk,
            require_atlas_on_disk=not args.no_check_disk,
            repo_root=repo_root,
            randomized=args.randomized_slate,
            rng=rng,
            atlas_embedding_cache_dir=(
                args.atlas_embedding_cache if args.randomized_slate else None
            ),
        ):
            payload: dict[str, Any] = {
                "section_id": state.section_id,
                "subject_id": state.subject_id,
                "atlas_name": state.atlas_name,
                "plane": state.plane,
                "dataset": state.dataset,
                "valid_range_mm": list(state.valid_range_mm),
                "ground_truth_mm": state.ground_truth_mm,
                "query_image_path": state.query_image_path,
                "atlas_slate_paths": list(state.atlas_slate_paths),
                "atlas_slate_positions_mm": list(state.atlas_slate_positions_mm),
                "difficulty_score": state.difficulty_score,
                "source": state.source,
                "quality": state.quality if state.quality is not None else {},
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n_total += 1
            if state.source == RANDOMIZED_LANE_B_SOURCE:
                n_synthesized += 1

    n_deterministic = n_total - n_synthesized
    print(
        f"[section_state] total written: {n_total} rows "
        f"(synthesized={n_synthesized}, deterministic={n_deterministic}) "
        f"to {args.output}",
        file=sys.stderr, flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="single_turn_rl.section_state",
        description="Build index-driven SectionState JSONL for single-turn GRPO (Lane B).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser(
        "build",
        help="Walk the manifest index and emit SectionState JSONL.",
    )
    pb.add_argument("--manifest-root", type=Path, default=Path("data/manifest"),
                    help="Root of the data manifest. Defaults to data/manifest "
                    "(canonical per AGENTS.md).")
    pb.add_argument("--output", type=Path, required=True,
                    help="Destination JSONL path.")
    pb.add_argument("--plane", type=str, default="coronal",
                    help="Plane filter (coronal/sagittal/horizontal). "
                    "Default: coronal.")
    pb.add_argument("--atlas", type=str, default="allen_mouse_25um",
                    help="Atlas filter. Default: allen_mouse_25um.")
    pb.add_argument("--split", type=str, default="rlvr",
                    help="Allocation split filter (rlvr/sft/eval/'' for any). "
                    "Default: rlvr.")
    pb.add_argument("--slate-root", type=Path, default=None,
                    help="Root for canonical-slate JSON cache. Defaults to "
                    "<repo_root>/models/langslice-gemma-4/data/.")
    pb.add_argument("--n-positions", type=int, default=9,
                    help="Number of positions in the canonical slate (deterministic "
                    "path only). Default: 9.")
    pb.add_argument("--no-check-disk", action="store_true",
                    help="Skip image-on-disk existence checks.")
    pb.add_argument("--repo-root", type=Path, default=None,
                    help="Repo root for resolving repo-relative paths. "
                    "Defaults to the current working directory.")
    pb.add_argument("--randomized-slate", action="store_true",
                    help="Replace the deterministic canonical slate with a "
                    "fresh per-row randomized Lane B slate from "
                    "langslice_traces.generate_trace(strategy='lane_b_broad_slate'). "
                    "Synthetic rows are tagged source='procedural_generator:lane_b'. "
                    "Off by default; existing pipelines see no change unless set.")
    pb.add_argument("--randomized-seed", type=int, default=1337,
                    help="Seed for the randomized-slate random.Random. "
                    "Same seed yields identical slates across reruns. "
                    "Default: 1337.")
    pb.add_argument("--atlas-embedding-cache", type=Path,
                    default=Path("out/atlas_embeddings"),
                    help="Directory containing <atlas>_<plane>.pt embedding "
                    "caches. Used by --randomized-slate to load the per-plane "
                    "grid of valid atlas positions. Default: out/atlas_embeddings.")
    pb.set_defaults(func=_build_cmd)

    ns = p.parse_args(argv)
    return int(ns.func(ns))


__all__ = [
    "RANDOMIZED_LANE_A_SOURCE",
    "RANDOMIZED_LANE_B_SOURCE",
    "CanonicalSlate",
    "SectionState",
    "build_canonical_slate",
    "build_randomized_section_state",
    "get_or_build_canonical_slate",
    "iter_section_states",
    "load_canonical_slate",
    "save_canonical_slate",
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main(sys.argv[1:]))
