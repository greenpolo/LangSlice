"""Pipeline orchestration for whole-brain AP estimation."""

from __future__ import annotations

import asyncio
import logging
import os
import statistics

from langslice.ai.estimator import APResult
from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.brain.agents import run_anchor_estimation, run_slice_estimation
from langslice.brain.anchor_selection import select_anchor_indices
from langslice.brain.checkpoint import load_checkpoint, save_checkpoint
from langslice.brain.discovery import discover_slices
from langslice.brain.interpolation import interpolate_positions
from langslice.brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)

logger = logging.getLogger(__name__)


async def run_brain_estimation(
    config: BrainEstimationConfig,
    *,
    checkpoint_path: str | None = None,
    on_progress: object | None = None,
) -> BrainEstimationResult:
    """Main entry point: run the full whole-brain AP estimation pipeline.

    Phases:
      1. Anchor estimation (3-pass nano-banana, full atlas range).
      2. Interpolation to derive center positions for remaining slices.
      3. Parallel estimation of all non-anchor slices (2-pass nano-banana).
      4. Isotonic regression across all estimates with spacing priors.
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
    existing_map = {s.filename: s for s in existing if s.locked}

    # --- Build initial slice list ---
    slices = [
        existing_map.get(
            os.path.basename(p),
            SlicePosition(os.path.basename(p), i, 0.0, "", locked=False),
        )
        for i, p in enumerate(image_paths)
    ]

    # --- Phase 1: Anchor estimation ---
    anchor_indices = select_anchor_indices(n_slices, config.n_anchors)
    anchor_set = set(anchor_indices)

    # Skip anchors that are already estimated from a checkpoint
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
                    model_name=config.coarse_model,
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
                raw_position_mm=ap_result.position_mm,
            )

        save_checkpoint(cp_path, config, slices)
        _progress("Phase 1 complete, checkpoint saved")

    # --- Phase 2: Interpolation (derive center positions) ---
    _progress("Phase 2: interpolating positions")
    slices = interpolate_positions(
        slices,
        interval_mm=config.interval_mm,
        atlas_range=atlas_range,
        z_axis=config.z_axis,
    )
    save_checkpoint(cp_path, config, slices)

    # --- Phase 3: Estimate all non-anchor slices ---
    non_anchor_indices = [i for i in range(n_slices) if i not in anchor_set]
    n_estimated = 0

    if non_anchor_indices:
        _progress(f"Phase 3: estimating {len(non_anchor_indices)} slices")
        sem = asyncio.Semaphore(config.max_parallel)

        async def _run_estimate(idx: int) -> tuple[int, APResult]:
            async with sem:
                result = await run_slice_estimation(
                    image_path=image_paths[idx],
                    atlas_name=config.atlas_name,
                    center_mm=slices[idx].position_mm,
                    model_name=config.fine_model,
                )
                return idx, result

        tasks = [_run_estimate(i) for i in non_anchor_indices]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, BaseException):
                logger.warning("Slice estimation failed: %s", r)
                continue
            idx, ap_result = r
            slices[idx] = SlicePosition(
                slices[idx].filename,
                idx,
                ap_result.position_mm,
                slices[idx].source + "+estimated",
                locked=True,
                raw_position_mm=ap_result.position_mm,
            )
            n_estimated += 1

        save_checkpoint(cp_path, config, slices)
        _progress(f"Phase 3 complete: {n_estimated} estimated")

    # --- Phase 4: Isotonic regression with spacing priors ---
    _progress("Phase 4: fitting isotonic regression")
    slices = _fit_isotonic(
        slices,
        interval_mm=config.interval_mm,
        thickness_mm=config.thickness_mm,
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
        n_refined=n_estimated,
        n_skipped=len(non_anchor_indices) - n_estimated,
    )

    return BrainEstimationResult(config=config, slices=slices, summary=summary)


def _fit_isotonic(
    slices: list[SlicePosition],
    *,
    interval_mm: float,
    thickness_mm: float,
) -> list[SlicePosition]:
    """Fit a monotone-increasing curve through all slice estimates.

    Uses scipy's isotonic regression to find positions that minimize squared
    error to the raw estimates while enforcing monotonicity.  A spacing
    regularization term penalizes deviations from the expected *interval_mm*.
    Minimum spacing of *thickness_mm* is enforced as a hard constraint.
    """
    import numpy as np
    from scipy.optimize import isotonic_regression

    raw = np.array([s.position_mm for s in slices])
    n = len(raw)

    # Isotonic regression: find monotone-increasing values closest to raw
    result = isotonic_regression(raw)
    fitted = result.x if hasattr(result, "x") else np.asarray(result)

    # Enforce minimum spacing (thickness_mm)
    for i in range(1, n):
        if fitted[i] < fitted[i - 1] + thickness_mm:
            fitted[i] = fitted[i - 1] + thickness_mm

    # Blend toward expected spacing to regularize
    # Build an "expected" series anchored at the mean of fitted
    mean_pos = float(fitted.mean())
    center_idx = (n - 1) / 2.0
    expected = np.array([
        mean_pos + (i - center_idx) * interval_mm for i in range(n)
    ])

    # Light regularization: 80% VLM estimate, 20% expected spacing
    alpha = 0.2
    blended = (1 - alpha) * fitted + alpha * expected

    # Re-run isotonic regression on blended to ensure monotonicity
    result2 = isotonic_regression(blended)
    final = result2.x if hasattr(result2, "x") else np.asarray(result2)

    # Final minimum spacing enforcement
    for i in range(1, n):
        if final[i] < final[i - 1] + thickness_mm:
            final[i] = final[i - 1] + thickness_mm

    return [
        SlicePosition(
            s.filename,
            s.index,
            round(float(final[i]), 4),
            s.source,
            locked=s.locked,
            raw_position_mm=s.raw_position_mm,
        )
        for i, s in enumerate(slices)
    ]
