"""Procedural trace generator: bare allocation row -> fabricated CanonicalTrace prefix.

Unlike parser.py which consumes existing Gemini-authored traces, this module
synthesizes a trace from scratch using the pre-embedded atlas grid as the
universe of fetchable positions. Output traces have ``final_answer=None`` -- the
terminal submit_estimate is the consumer's job (Gemma at inference time).

Two strategies, both sampling from the frozen empirical distributions in
:mod:`langslice_traces._empirical` so the generated traces are statistically
indistinguishable from real Gemini behavior at cohort scale:

* ``"lane_a_prefix"``: multi-step prefix (Lane A & iSFT). Step 0 is a broad
  GT-anchored sweep; step 1+ is a narrow fine bracket with the GT inside but
  rarely centered. Positions are snapped through a roundness ladder
  (int/half/tenth/raw) matching the corpus.
* ``"lane_b_broad_slate"``: single broad slate with GT inside the range but
  randomly off-center. Width clamped to a meaningfully broad band.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

from ._empirical import (
    LANE_B_EDGE_GUARD,
    LANE_B_WIDTH_HI_MM,
    LANE_B_WIDTH_LO_MM,
    P_CENTER_OFFSET_STEP1,
    P_N_TOOL_STEPS,
    P_NFETCH_LANE_B,
    P_NFETCH_STEP0,
    P_NFETCH_STEP1,
    P_ROUNDNESS_STEP0,
    P_ROUNDNESS_STEP1,
    P_SPAN_STEP0,
    P_SPAN_STEP1,
    SIGMA_ANCHOR_MM,
    sample,
    sample_roundness,
)
from .augmentations import _snap_to_grid
from .schema import CanonicalTrace, Plane, ToolStep

GenerationStrategy = Literal["lane_a_prefix", "lane_b_broad_slate"]

_PLANE_AXIS: dict[Plane, str] = {"coronal": "AP", "sagittal": "ML", "horizontal": "DV"}

# Roundness ladder ordered coarsest -> finest. Two uses:
# (1) Fallback when a chosen tier has no candidate grid points in range
#     (rare on a dense grid but possible near plane edges or on sparse grids).
# (2) Collision avoidance in :func:`_snap_positions_with_roundness`: if a
#     tier-snapped position collides with an earlier position in the same
#     fetch, step to the next finer tier and retry.
_TIER_FALLBACK: list[str] = ["int", "half", "tenth", "raw"]

# Small margin (mm) applied when clipping the step-0 anchor to plane bounds.
# Keeps the fine sweep from being squeezed against a hard wall.
_ANCHOR_CLIP_MARGIN_MM: float = 0.05


def _user_prompt(plane: Plane, atlas_name: str) -> str:
    """Build the user prompt matching the existing SFT corpus template."""
    axis = _PLANE_AXIS[plane]
    return f"Determine this {plane} slice's {axis} position in the {atlas_name} atlas."


def _format_position(p: float) -> str:
    """Format a position as ``"<X.XX>"`` (two-decimal mm)."""
    return f"{p:.2f}"


def _image_path(atlas_name: str, plane: Plane, position_mm: float) -> str:
    """Build the tool-result image relpath ``"atlas/<atlas>/<plane>/<p:.2f>mm.jpg"``.

    This is the *generator-native* form (bare ``atlas/...`` prefix). Downstream
    consumers that need a repo-relative path the trainer can resolve under
    ``repo_root`` should pass each path through
    :func:`canonical_atlas_repo_path` at row-construction time.
    """
    return f"atlas/{atlas_name}/{plane}/{_format_position(position_mm)}mm.jpg"


# Repo-relative prefix where the canonical atlas tiles actually live on disk.
# Kept as a module-level constant so both the helper below and consumers that
# import it directly stay in sync if the layout ever moves.
CANONICAL_ATLAS_ROOT: str = "models/langslice-gemma-4/data/atlas"
_CANONICAL_ATLAS_ROOT_ENV = "LANGSLICE_CANONICAL_ATLAS_ROOT"


def canonical_atlas_root() -> str:
    """Return canonical atlas root, allowing env override for new layouts."""
    return os.environ.get(_CANONICAL_ATLAS_ROOT_ENV, CANONICAL_ATLAS_ROOT)


def canonical_atlas_repo_path(generator_path: str) -> str:
    """Translate any of the known atlas-path prefixes to the canonical
    repo-relative form ``models/langslice-gemma-4/data/atlas/<atlas>/<plane>/<X.XX>mm.jpg``.

    The trainer's image-resolution rule is ``repo_root / p`` (see
    ``single_turn_rl/dataset.py:177``); the on-disk atlas tiles live under
    ``models/langslice-gemma-4/data/atlas/``, so every consumer of generator
    output must rewrite paths to that canonical form before handing them
    to the trainer.

    Three input shapes are accepted (and forward / back slashes both work):

    * Already-canonical: ``models/langslice-gemma-4/data/atlas/...`` →
      returned unchanged (helper is **idempotent**).
    * Generator-native: ``atlas/<atlas>/<plane>/<X.XX>mm.jpg`` → the canonical
      root is prepended.
    * Legacy SectionState: ``data/atlas/<atlas>/<plane>/<X.XX>mm.jpg`` → the
      ``data/`` segment is rewritten to the canonical root so old Lane B
      JSONL files keep loading.

    Anything else (already-absolute path, or a string that doesn't start with
    one of the above prefixes) is returned unchanged — the helper is a
    rewriter, not a validator.
    """
    s = str(generator_path).replace("\\", "/")
    canonical_prefix = canonical_atlas_root().rstrip("/") + "/"
    if s.startswith(canonical_prefix):
        return s
    if s.startswith("data/atlas/"):
        return canonical_prefix + s[len("data/atlas/"):]
    if s.startswith("atlas/"):
        return canonical_prefix + s[len("atlas/"):]
    return s


def _result_text(positions: list[float]) -> str:
    """Build the ``"Fetched N atlas section(s): ..."`` text matching the corpus.

    Singular ``"1 atlas section"`` when n=1; ``"N atlas sections"`` otherwise.
    Positions are rendered ``"{:.2f} mm"`` with a space before ``mm``.
    """
    n = len(positions)
    noun = "atlas section" if n == 1 else "atlas sections"
    formatted = ", ".join(f"{_format_position(p)} mm" for p in positions)
    return f"Fetched {n} {noun}: {formatted}"


def _dedupe_preserving_order(values: list[float]) -> list[float]:
    """Drop duplicates while preserving first-occurrence order."""
    seen: set[float] = set()
    out: list[float] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _build_tool_step(
    positions: list[float],
    *,
    atlas_name: str,
    plane: Plane,
) -> ToolStep:
    """Construct a fetch_atlas ToolStep from a position list.

    Caller is responsible for snapping/deduping; this helper only formats the
    images + text. ``call_args.positions_mm`` is stored as the actual list
    passed in (already snapped, already deduped).
    """
    return ToolStep(
        call_name="fetch_atlas",
        call_args={"positions_mm": list(positions)},
        result_image_paths=[_image_path(atlas_name, plane, p) for p in positions],
        result_text=_result_text(positions),
    )


def load_atlas_grid(cache_dir: Path, atlas_name: str, plane: Plane) -> list[float]:
    """Read ``<cache_dir>/<atlas_name>_<plane>.pt`` and return sorted positions (mm).

    Basenames in the pre-embedded cache look like ``"3.20mm.jpg"``; parses
    them and returns the sorted float list. Raises FileNotFoundError if the
    .pt file is absent.
    """
    import torch  # local import: keeps the rest of the module torch-free

    cache_path = Path(cache_dir) / f"{atlas_name}_{plane}.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"atlas embedding cache not found: {cache_path}")
    payload = torch.load(cache_path, weights_only=False)
    embeddings = payload["embeddings"]
    positions: list[float] = []
    for basename in embeddings.keys():
        name = str(basename)
        idx = name.find("mm")
        if idx <= 0:
            raise ValueError(
                f"unexpected basename in embeddings cache: {name!r} (expected '<float>mm.<ext>')"
            )
        try:
            positions.append(float(name[:idx]))
        except ValueError as exc:
            raise ValueError(
                f"unable to parse position from basename {name!r}"
            ) from exc
    positions.sort()
    return positions


def _is_tier_match(value: float, tier: str) -> bool:
    """Return True if ``value`` belongs to the given roundness tier.

    Tiers
    -----
    * ``"int"``   - value is a whole millimeter (e.g. 3.0, 5.0).
    * ``"half"``  - value is a multiple of 0.5 (covers ints too).
    * ``"tenth"`` - value is a multiple of 0.1 (covers ints and halves).
    * ``"raw"``   - no constraint; every grid point qualifies.

    Note the tiers are nested: every integer is also a half-multiple and a
    tenth-multiple. That's intentional. :func:`_filter_grid_by_tier` uses
    these checks *additively* -- tier "half" returns ints + halves, so a
    "half" draw naturally lands on whole-mm positions roughly half the time.
    Tier "int" still excludes non-integers because only the int check passes
    for those values.
    """
    if tier == "raw":
        return True
    if tier == "int":
        return abs(value - round(value)) < 1e-6
    if tier == "half":
        return abs(value * 2 - round(value * 2)) < 1e-6
    if tier == "tenth":
        return abs(value * 10 - round(value * 10)) < 1e-6
    raise ValueError(f"unknown roundness tier: {tier!r}")


def _filter_grid_by_tier(grid: list[float], tier: str) -> list[float]:
    """Return the subset of ``grid`` matching the roundness tier.

    The filter is *additive*: tier "half" returns ints + halves, tier "tenth"
    returns ints + halves + tenths, etc. This matches how a Gemini agent
    actually picks round-mm fetches -- "I want a half-mm" naturally includes
    whole-mm positions.
    """
    return [g for g in grid if _is_tier_match(g, tier)]


def _snap_with_tier(value: float, grid: list[float], tier: str) -> float:
    """Snap ``value`` to the nearest grid point matching ``tier``.

    Falls back to progressively finer tiers if the requested tier has no
    candidates -- this happens on very sparse grids or near plane edges. In
    practice the dense pre-embedded grids never trigger fallback.
    """
    if not grid:
        raise ValueError("grid must be non-empty")
    start = _TIER_FALLBACK.index(tier) if tier in _TIER_FALLBACK else 0
    for t in _TIER_FALLBACK[start:]:
        candidates = _filter_grid_by_tier(grid, t)
        if candidates:
            return min(candidates, key=lambda g: abs(g - value))
    # The last tier is "raw" which always returns the full grid (non-empty by
    # the check above), so this fallback path is unreachable. Keep the
    # exception for safety.
    return _snap_to_grid(value, grid)


def _place_evenly(lo: float, hi: float, n: int) -> list[float]:
    """Return ``n`` evenly-spaced positions in ``[lo, hi]`` (endpoints inclusive)."""
    if n <= 0:
        return []
    if n == 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + step * k for k in range(n)]


def _clip(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return max(lo, min(hi, value))


def _snap_positions_with_roundness(
    raw_positions: list[float],
    grid: list[float],
    rng: random.Random,
    roundness_dist: dict[str, float],
) -> list[float]:
    """Snap each raw position through a per-position roundness draw.

    Each position is independently assigned a roundness tier (int/half/tenth/
    raw) and snapped to the nearest grid point in that tier. This reproduces
    the real corpus where individual fetches in the same step can sit on
    different roundness tiers.

    Collision avoidance: if the initial tier draw would snap a position onto
    a grid point already used by an earlier position in the same fetch, retry
    with progressively finer tiers until a free grid point is found. Without
    this the n_fetch distribution drifts off the empirical post-snap count
    (a 1mm-span n=5 fetch on integer tier would collapse to 2 positions).
    The real agent implicitly picks tiers that avoid collisions; we mirror
    that without losing the per-position tier diversity.
    """
    used: set[float] = set()
    snapped: list[float] = []
    for raw in raw_positions:
        tier = sample_roundness(rng, roundness_dist)
        start = _TIER_FALLBACK.index(tier) if tier in _TIER_FALLBACK else 0
        picked: float | None = None
        for t in _TIER_FALLBACK[start:]:
            value = _snap_with_tier(raw, grid, t)
            if value not in used:
                picked = value
                break
        if picked is None:
            # Every tier collided -- accept the raw-tier snap even on collision
            # so the caller's dedupe still keeps the position list ordered.
            picked = _snap_with_tier(raw, grid, "raw")
        used.add(picked)
        snapped.append(picked)
    return snapped


def _lane_a_prefix(
    *,
    ground_truth_mm: float,
    grid: list[float],
    atlas_name: str,
    plane: Plane,
    rng: random.Random,
) -> tuple[list[ToolStep], dict]:
    """Build the Lane A / iSFT prefix from the empirical distributions.

    Returns ``(tool_steps, quality)`` so the caller can attach quality fields.
    """
    plane_lo = grid[0]
    plane_hi = grid[-1]
    anchor_lo = plane_lo + _ANCHOR_CLIP_MARGIN_MM
    anchor_hi = plane_hi - _ANCHOR_CLIP_MARGIN_MM
    # If the grid is so short the margin inverts, just collapse to grid endpoints.
    if anchor_lo >= anchor_hi:
        anchor_lo, anchor_hi = plane_lo, plane_hi

    n_steps = sample(rng, P_N_TOOL_STEPS)
    tool_steps: list[ToolStep] = []

    for step_idx in range(n_steps):
        if step_idx == 0:
            # Step 0: GT-anchored broad sweep, image-informed but not GT-leaky.
            anchor = ground_truth_mm + rng.gauss(0.0, SIGMA_ANCHOR_MM)
            anchor = _clip(anchor, anchor_lo, anchor_hi)
            span = sample(rng, P_SPAN_STEP0)
            n_fetch = sample(rng, P_NFETCH_STEP0)
            raw_lo = _clip(anchor - span / 2.0, plane_lo, plane_hi)
            raw_hi = _clip(anchor + span / 2.0, plane_lo, plane_hi)
            raw = _place_evenly(raw_lo, raw_hi, n_fetch)
            snapped = _snap_positions_with_roundness(raw, grid, rng, P_ROUNDNESS_STEP0)
        else:
            # Step 1+: narrow fine bracket centered near (but not exactly at) GT.
            offset = sample(rng, P_CENTER_OFFSET_STEP1)
            span = sample(rng, P_SPAN_STEP1)
            n_fetch = sample(rng, P_NFETCH_STEP1)
            center = ground_truth_mm + offset
            raw_lo = _clip(center - span / 2.0, plane_lo, plane_hi)
            raw_hi = _clip(center + span / 2.0, plane_lo, plane_hi)
            raw = _place_evenly(raw_lo, raw_hi, n_fetch)
            snapped = _snap_positions_with_roundness(raw, grid, rng, P_ROUNDNESS_STEP1)

        deduped = _dedupe_preserving_order(snapped)
        # Real corpus n=1 steps are vanishingly rare and bracket-shaped; drop
        # any step that dedupes to <2 positions to keep the realism profile.
        if len(deduped) < 2:
            continue
        # Per-position roundness draws can place a later raw value onto an
        # earlier-snapped tier and produce a smaller mm than its predecessor,
        # leaving positions unsorted. Real corpus is essentially 100% sorted
        # (0.03% unsorted), so we sort here to match.
        sorted_positions = sorted(deduped)
        tool_steps.append(_build_tool_step(sorted_positions, atlas_name=atlas_name, plane=plane))

    quality = {"source": "procedural_generator", "strategy": "lane_a_prefix"}
    return tool_steps, quality


def _lane_b_broad_slate(
    *,
    ground_truth_mm: float,
    grid: list[float],
    atlas_name: str,
    plane: Plane,
    rng: random.Random,
) -> tuple[list[ToolStep], dict]:
    """Build the Lane B single-slate prefix from the empirical distributions.

    Returns ``(tool_steps, quality)`` so the caller can stash the realized
    gt_fraction_in_slate on the trace.
    """
    plane_lo = grid[0]
    plane_hi = grid[-1]

    width = _clip(sample(rng, P_SPAN_STEP0), LANE_B_WIDTH_LO_MM, LANE_B_WIDTH_HI_MM)

    def _sample_fraction() -> float:
        return rng.uniform(0.1, 0.9)

    def _bracket_for(frac: float) -> tuple[float, float, float]:
        """Return (lo, hi, realized_gt_fraction) after clipping to plane bounds."""
        lo = ground_truth_mm - frac * width
        hi = ground_truth_mm + (1.0 - frac) * width
        # Clip to plane bounds while preserving the requested width when
        # possible. If one edge clips, slide the slate so the width is
        # preserved -- this keeps Lane B genuinely broad even near plane ends.
        if lo < plane_lo:
            shift = plane_lo - lo
            lo += shift
            hi += shift
        if hi > plane_hi:
            shift = hi - plane_hi
            lo -= shift
            hi -= shift
        lo = max(lo, plane_lo)
        hi = min(hi, plane_hi)
        realized_width = hi - lo
        if realized_width <= 0:
            return lo, hi, 0.5
        realized_frac = (ground_truth_mm - lo) / realized_width
        return lo, hi, realized_frac

    frac = _sample_fraction()
    lo, hi, realized = _bracket_for(frac)
    if realized < LANE_B_EDGE_GUARD or realized > (1.0 - LANE_B_EDGE_GUARD):
        # Redraw once if the clip pushed GT to a slate edge.
        frac = _sample_fraction()
        lo, hi, realized = _bracket_for(frac)

    n_positions = sample(rng, P_NFETCH_LANE_B)
    raw = _place_evenly(lo, hi, n_positions)
    snapped = _snap_positions_with_roundness(raw, grid, rng, P_ROUNDNESS_STEP0)
    deduped = _dedupe_preserving_order(snapped)

    # Hard contract: the slate brackets GT. Roundness snapping can occasionally
    # collapse the boundary positions inward of GT (e.g. raw_hi=4.1 snapped to
    # int tier -> 4.0, with GT=4.05). When this happens, rescue by inserting
    # the nearest grid point on the missing side -- the rescue is rare on
    # dense grids and keeps the slate genuinely bracketing the GT.
    if min(deduped) > ground_truth_mm:
        # Find the largest grid point at-or-below GT and prepend it.
        below = [g for g in grid if g <= ground_truth_mm]
        if below:
            deduped = [max(below), *deduped]
    if max(deduped) < ground_truth_mm:
        # Find the smallest grid point at-or-above GT and append it.
        above = [g for g in grid if g >= ground_truth_mm]
        if above:
            deduped = [*deduped, min(above)]
    deduped = _dedupe_preserving_order(deduped)

    # Recompute realized fraction after any rescue insertion.
    slate_lo, slate_hi = min(deduped), max(deduped)
    if slate_hi > slate_lo:
        realized = (ground_truth_mm - slate_lo) / (slate_hi - slate_lo)
    else:
        realized = 0.5

    if not (slate_lo <= ground_truth_mm <= slate_hi):
        raise AssertionError(
            "lane_b_broad_slate slate failed to bracket GT after rescue: "
            f"gt={ground_truth_mm} positions={deduped}"
        )

    # Per-position roundness draws (and rescue insertions) can leave positions
    # out of order. Real corpus is essentially 100% sorted (0.03% unsorted),
    # so we sort here to match.
    sorted_positions = sorted(deduped)

    tool_steps = [_build_tool_step(sorted_positions, atlas_name=atlas_name, plane=plane)]
    quality = {
        "source": "procedural_generator",
        "strategy": "lane_b_broad_slate",
        "gt_fraction_in_slate": float(realized),
    }
    return tool_steps, quality


def generate_trace(
    *,
    image_path: str,
    ground_truth_mm: float,
    plane: Plane,
    atlas_name: str,
    atlas_version: str,
    subject_id: str,
    grid: list[float],
    strategy: GenerationStrategy,
    rng: random.Random,
    bucket: int = 1,  # corpus is 100% bucket=1; can be overridden
) -> CanonicalTrace:
    """Synthesize a prefix-only CanonicalTrace from the empirical distributions.

    Strategies
    ----------
    ``"lane_a_prefix"``:
        Multi-step prefix matching Gemini's real behavior. Step 0 is a broad
        GT-anchored sweep (anchor at ``gt + N(0, SIGMA_ANCHOR_MM)``, span and
        n_fetch from :data:`P_SPAN_STEP0` / :data:`P_NFETCH_STEP0`). Step 1+
        is a narrow fine bracket (center at ``gt + offset`` where offset is
        from :data:`P_CENTER_OFFSET_STEP1`). Per-position roundness draws snap
        each fetch to the corpus-matched ladder (int/half/tenth/raw).

    ``"lane_b_broad_slate"``:
        Single broad slate. Width drawn from :data:`P_SPAN_STEP0` clamped to
        ``[3, 12]`` mm; GT-fraction-in-slate sampled uniformly from
        ``[0.1, 0.9]`` (so GT is inside but rarely centered). 7-9 evenly
        spaced positions, snapped via the step-0 roundness ladder.
    """
    if not grid:
        raise ValueError("grid must be non-empty")

    if strategy == "lane_a_prefix":
        tool_steps, quality = _lane_a_prefix(
            ground_truth_mm=ground_truth_mm,
            grid=grid,
            atlas_name=atlas_name,
            plane=plane,
            rng=rng,
        )
    elif strategy == "lane_b_broad_slate":
        tool_steps, quality = _lane_b_broad_slate(
            ground_truth_mm=ground_truth_mm,
            grid=grid,
            atlas_name=atlas_name,
            plane=plane,
            rng=rng,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    return CanonicalTrace(
        atlas_name=atlas_name,
        atlas_version=atlas_version,
        plane=plane,
        subject_id=subject_id,
        system_prompt_kind="single_slice",
        bucket=bucket,
        query_image_paths=[image_path],
        user_prompt_text=_user_prompt(plane, atlas_name),
        tool_steps=tool_steps,
        final_answer=None,
        quality=quality,
        gemini_reasoning=None,
        dataset_root=None,
    )


if __name__ == "__main__":  # pragma: no cover
    """Smoke print of a handful of traces for visual inspection."""
    rng = random.Random(0)
    grid = [round(i * 0.1, 1) for i in range(101)]  # 0.0 .. 10.0 step 0.1
    for strat in ("lane_a_prefix", "lane_b_broad_slate"):
        print(f"\n=== strategy={strat} ===")
        for k in range(3):
            t = generate_trace(
                image_path=f"queries/img_{k}.jpg",
                ground_truth_mm=rng.uniform(2.0, 8.0),
                plane="coronal",
                atlas_name="allen_mouse_25um",
                atlas_version="CCFv3",
                subject_id=f"M{k:02d}",
                grid=grid,
                strategy=strat,  # type: ignore[arg-type]
                rng=rng,
            )
            print(f"trace {k}: quality={t.quality}")
            for i, step in enumerate(t.tool_steps):
                print(f"  step {i}: {step.call_args['positions_mm']}")
