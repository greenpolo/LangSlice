"""Wave computation and pipeline orchestration for whole-brain estimation."""

from __future__ import annotations

import asyncio
import logging
import os
import statistics

from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.ai.estimator import APResult
from langslice.brain.agents import run_anchor_estimation, run_refinement
from langslice.brain.anchor_selection import select_anchor_indices
from langslice.brain.checkpoint import load_checkpoint, save_checkpoint
from langslice.brain.constraints import enforce_constraints
from langslice.brain.discovery import discover_slices
from langslice.brain.interpolation import interpolate_positions
from langslice.brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)
from langslice.brain.window import compute_refinement_window

logger = logging.getLogger(__name__)


def compute_waves(n_slices: int, anchor_indices: set[int]) -> list[list[int]]:
    """Compute refinement waves radiating outward from anchors.

    Returns a list of waves, where each wave is a list of slice indices that
    can be processed in parallel.  Slices at distance 1 from any locked index
    are in wave 0, distance 2 in wave 1, etc.
    """
    remaining = set(range(n_slices)) - anchor_indices
    locked = set(anchor_indices)
    waves: list[list[int]] = []

    while remaining:
        wave: list[int] = []
        for idx in sorted(remaining):
            if (idx - 1) in locked or (idx + 1) in locked:
                wave.append(idx)
        if not wave:
            # Unreachable if anchors exist, but safety fallback
            wave = sorted(remaining)
        for idx in wave:
            remaining.discard(idx)
        locked.update(wave)
        waves.append(wave)

    return waves


async def run_brain_estimation(
    config: BrainEstimationConfig,
    *,
    checkpoint_path: str | None = None,
    on_progress: object | None = None,
) -> BrainEstimationResult:
    """Main entry point: run the full whole-brain AP estimation pipeline.

    Phases:
      1. Parallel anchor estimation (coarse + nano-banana).
      2. Deterministic interpolation.
      3. Wave-based nano-banana refinement (optional).
      4. Constraint enforcement.
    """

    def _progress(msg: str) -> None:
        if callable(on_progress):
            on_progress(msg)
        logger.info(msg)

    # --- Discover images ---
    image_paths = discover_slices(config.image_folder)
    n_slices = len(image_paths)
    if n_slices == 0:
        raise ValueError(f"No images found in {config.image_folder}")
    _progress(f"Found {n_slices} slices")

    # --- Load atlas for bounds ---
    atlas = load_atlas(config.atlas_name)
    atlas_range = get_position_range_mm(atlas)

    # --- Check for checkpoint ---
    cp_path = checkpoint_path or os.path.join(config.image_folder, "brain_estimate.json")
    existing = load_checkpoint(cp_path)
    existing_locked = {s.filename: s for s in existing if s.locked}

    # --- Build initial slice list ---
    slices = [
        existing_locked.get(
            os.path.basename(p),
            SlicePosition(os.path.basename(p), i, 0.0, "", locked=False),
        )
        for i, p in enumerate(image_paths)
    ]

    # --- Phase 1: Anchor estimation ---
    anchor_indices = select_anchor_indices(n_slices, config.n_anchors)
    anchor_set = set(anchor_indices)

    # Skip anchors that are already locked from a checkpoint
    anchors_to_run = [i for i in anchor_indices if not slices[i].locked]

    if anchors_to_run:
        _progress(f"Phase 1: estimating {len(anchors_to_run)} anchors")
        sem = asyncio.Semaphore(config.max_parallel)

        async def _run_anchor(idx: int) -> tuple[int, APResult]:
            async with sem:
                _progress(f"  Anchor slice {idx} ({os.path.basename(image_paths[idx])})")
                result = await run_anchor_estimation(
                    image_path=image_paths[idx],
                    atlas_name=config.atlas_name,
                )
                return idx, result

        tasks = [_run_anchor(i) for i in anchors_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, BaseException):
                raise RuntimeError(f"Anchor estimation failed: {r}") from r
            idx, ap_result = r
            slices[idx] = SlicePosition(
                slices[idx].filename,
                idx,
                ap_result.position_mm,
                "anchor",
                locked=True,
            )

        # Sanity check: monotonic order
        anchor_positions = [(i, slices[i].position_mm) for i in anchor_indices]
        _validate_anchor_order(anchor_positions, config.z_axis)

        save_checkpoint(cp_path, config, slices)
        _progress("Phase 1 complete, checkpoint saved")

    # --- Phase 2: Interpolation ---
    _progress("Phase 2: interpolating positions")
    slices = interpolate_positions(
        slices,
        interval_mm=config.interval_mm,
        atlas_range=atlas_range,
        z_axis=config.z_axis,
    )
    save_checkpoint(cp_path, config, slices)

    # --- Phase 3: Nano-banana refinement ---
    if config.refinement:
        waves = compute_waves(n_slices, anchor_set)
        _progress(f"Phase 3: {len(waves)} refinement waves")
        sem = asyncio.Semaphore(config.max_parallel)
        n_refined = 0
        n_skipped = 0

        for wave_num, wave in enumerate(waves):
            _progress(f"  Wave {wave_num + 1}/{len(waves)}: {len(wave)} slices")

            async def _run_refine(idx: int) -> tuple[int, APResult | None]:
                async with sem:
                    left_locked = _find_locked_neighbor(slices, idx, direction=-1)
                    right_locked = _find_locked_neighbor(slices, idx, direction=1)
                    win = compute_refinement_window(
                        position_mm=slices[idx].position_mm,
                        left_locked_mm=left_locked,
                        right_locked_mm=right_locked,
                        thickness_mm=config.thickness_mm,
                        interval_mm=config.interval_mm,
                    )
                    if win.skip:
                        return idx, None
                    return idx, await run_refinement(
                        image_path=image_paths[idx],
                        atlas_name=config.atlas_name,
                        window_lo=win.lo,
                        window_hi=win.hi,
                        window_center=win.center,
                        n_images=win.n_images,
                    )

            tasks = [_run_refine(i) for i in wave]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, BaseException):
                    logger.warning("Refinement failed for a slice: %s", r)
                    continue
                idx, ap_result = r
                if ap_result is None:
                    slices[idx] = SlicePosition(
                        slices[idx].filename,
                        idx,
                        slices[idx].position_mm,
                        slices[idx].source,
                        locked=True,
                    )
                    n_skipped += 1
                else:
                    slices[idx] = SlicePosition(
                        slices[idx].filename,
                        idx,
                        ap_result.position_mm,
                        slices[idx].source + "+refined",
                        locked=True,
                    )
                    n_refined += 1

            save_checkpoint(cp_path, config, slices)

        _progress(f"Phase 3 complete: {n_refined} refined, {n_skipped} skipped")
    else:
        n_refined = 0
        n_skipped = n_slices - len(anchor_indices)

    # --- Phase 4: Constraint enforcement ---
    _progress("Phase 4: enforcing constraints")
    slices = enforce_constraints(
        slices,
        ordering=config.ordering,
        thickness_mm=config.thickness_mm,
        z_axis=config.z_axis,
    )
    save_checkpoint(cp_path, config, slices)

    # --- Summary ---
    positions = [s.position_mm for s in slices]
    intervals = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]

    summary = BrainEstimationSummary(
        mean_interval_mm=statistics.mean(intervals) if intervals else 0.0,
        std_interval_mm=statistics.stdev(intervals) if len(intervals) > 1 else 0.0,
        n_slices=n_slices,
        n_anchors=len(anchor_indices),
        n_refined=n_refined,
        n_skipped=n_skipped,
    )

    return BrainEstimationResult(config=config, slices=slices, summary=summary)


def _validate_anchor_order(
    anchors: list[tuple[int, float]], z_axis: str
) -> None:
    """Raise if anchors are not in expected monotonic order."""
    if len(anchors) < 2:
        return
    for i in range(len(anchors) - 1):
        idx_a, pos_a = anchors[i]
        idx_b, pos_b = anchors[i + 1]
        if z_axis == "AP" and pos_b <= pos_a:
            raise ValueError(
                f"Anchor at slice {idx_b} ({pos_b:.3f}mm) is not posterior to "
                f"slice {idx_a} ({pos_a:.3f}mm). Check images or re-run."
            )
        if z_axis == "PA" and pos_b >= pos_a:
            raise ValueError(
                f"Anchor at slice {idx_b} ({pos_b:.3f}mm) is not anterior to "
                f"slice {idx_a} ({pos_a:.3f}mm). Check images or re-run."
            )


def _find_locked_neighbor(
    slices: list[SlicePosition], idx: int, direction: int
) -> float | None:
    """Walk in *direction* (-1 or +1) from *idx* to find nearest locked slice."""
    i = idx + direction
    while 0 <= i < len(slices):
        if slices[i].locked:
            return slices[i].position_mm
        i += direction
    return None
