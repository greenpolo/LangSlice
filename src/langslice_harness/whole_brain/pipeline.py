"""Pipeline orchestration for whole-brain AP estimation."""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
from collections.abc import Callable

from langslice_harness.atlas.core import get_position_range_mm, load_atlas
from langslice_harness.estimation import APResult
from langslice_harness.whole_brain.anchor_selection import select_anchor_indices
from langslice_harness.whole_brain.checkpoint import load_checkpoint, save_checkpoint
from langslice_harness.whole_brain.discovery import discover_slices
from langslice_harness.whole_brain.estimation_agents import (
    run_anchor_estimation,
    run_slice_estimation,
)
from langslice_harness.whole_brain.interpolation import interpolate_positions
from langslice_harness.whole_brain.types import (
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
    on_progress: Callable[[str], None] | None = None,
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
        sem = asyncio.Semaphore(1)  # Sequential to avoid API rate limits

        async def _run_anchor(idx: int) -> tuple[int, APResult]:
            async with sem:
                _progress(f"  Anchor slice {idx} ({os.path.basename(image_paths[idx])})")
                result = await run_anchor_estimation(
                    image_path=image_paths[idx],
                    atlas_name=config.atlas_name,
                    coarse_model=config.coarse_model,
                    fine_model=config.fine_model,
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

    # Save interpolated positions for outlier detection in Phase 3.5.
    # Phase 3 overwrites these with VLM estimates, so we capture them now.
    interpolated_map: dict[int, float] = {
        i: slices[i].position_mm
        for i in range(n_slices)
        if not slices[i].locked
    }

    # --- Phase 3: Estimate all non-anchor slices ---
    # Skip slices already locked from a previous checkpoint.
    non_anchor_indices = [
        i for i in range(n_slices) if i not in anchor_set and not slices[i].locked
    ]
    n_estimated = 0

    if non_anchor_indices:
        _progress(f"Phase 3: estimating {len(non_anchor_indices)} slices")
        sem = asyncio.Semaphore(config.max_parallel)
        first_anchor_idx = anchor_indices[0]
        last_anchor_idx = anchor_indices[-1]

        async def _run_estimate(idx: int) -> tuple[int, APResult]:
            async with sem:
                edge_anchor_mm: float | None = None
                edge_side: str | None = None
                if slices[idx].source == "extrapolated":
                    if idx < first_anchor_idx:
                        edge_anchor_mm = slices[first_anchor_idx].position_mm
                        edge_side = "leading"
                    elif idx > last_anchor_idx:
                        edge_anchor_mm = slices[last_anchor_idx].position_mm
                        edge_side = "trailing"
                result = await run_slice_estimation(
                    image_path=image_paths[idx],
                    atlas_name=config.atlas_name,
                    center_mm=slices[idx].position_mm,
                    atlas_range=atlas_range,
                    edge_anchor_mm=edge_anchor_mm,
                    edge_side=edge_side,
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

    # --- Phase 3.5: Re-estimate outlier slices with anchor pipeline ---
    # Identify non-anchor slices whose Phase 3 VLM estimate deviates
    # significantly from the Phase 2 interpolated position.  Large deviations
    # indicate the Flash model misjudged the slice — re-estimating with the
    # anchor pipeline (Pro model, full atlas range, 2-stage coarse+fine)
    # produces anchor-quality accuracy for these problem slices.
    #
    # Unlike hyp 002 (Pro with constrained windows → mixed), this uses the
    # FULL atlas range, which is the key factor in Pro's 4-14x anchor
    # accuracy advantage.
    _OUTLIER_DEVIATION_MM = 0.4
    outlier_indices: list[int] = []
    for i in non_anchor_indices:
        raw = slices[i].raw_position_mm
        interp = interpolated_map.get(i)
        if raw is not None and interp is not None:
            deviation = abs(raw - interp)
            if deviation > _OUTLIER_DEVIATION_MM:
                outlier_indices.append(i)
                logger.info(
                    "  Outlier slice %d: raw=%.3f interp=%.3f dev=%.3f",
                    i, raw, interp, deviation,
                )

    if outlier_indices:
        _progress(
            f"Phase 3.5: re-estimating {len(outlier_indices)} outlier slices "
            f"with anchor pipeline (Pro model, full atlas range)"
        )
        n_reestimated = 0
        for idx in outlier_indices:
            try:
                old_raw = slices[idx].raw_position_mm
                interp_pos = interpolated_map.get(idx, 0.0)
                _progress(
                    f"  Re-estimating slice {idx} "
                    f"({os.path.basename(image_paths[idx])}): "
                    f"raw={old_raw:.3f} interp={interp_pos:.3f}"
                )
                reest = await run_anchor_estimation(
                    image_path=image_paths[idx],
                    atlas_name=config.atlas_name,
                    coarse_model=config.coarse_model,
                    fine_model=config.fine_model,
                )
                slices[idx] = SlicePosition(
                    slices[idx].filename,
                    idx,
                    reest.position_mm,
                    slices[idx].source + "+reestimated",
                    locked=True,
                    raw_position_mm=reest.position_mm,
                )
                _progress(
                    f"  Slice {idx}: {reest.position_mm:.3f}mm "
                    f"(was {old_raw:.3f}mm)"
                )
                n_reestimated += 1
            except Exception as exc:
                logger.warning(
                    "Outlier re-estimation failed for slice %d: %s", idx, exc,
                )
        save_checkpoint(cp_path, config, slices)
        _progress(f"Phase 3.5 complete: {n_reestimated} re-estimated")

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
    data_delta_mm: float = 1.0,
    interval_delta_mm: float = 0.2,
    spacing_weight: float = 0.0,
    anchor_weight: float = 5.0,
) -> list[SlicePosition]:
    """Fit a monotone-increasing curve through all slice estimates.

    Uses constrained optimization with Huber-loss data fidelity and hard
    monotonicity constraints (minimum spacing of *thickness_mm*).

    Anchor estimates receive *anchor_weight* times the data-fidelity
    penalty of non-anchor slices, reflecting their higher reliability
    from two-stage (coarse + fine) estimation.

    The Huber loss is quadratic for small residuals and linear for large
    ones, so accurate estimates are preserved while outliers are tamed.

    Parameters
    ----------
    data_delta_mm : float
        Huber threshold for data fidelity — residuals below this are
        penalized quadratically, above linearly.
    interval_delta_mm : float
        Huber threshold for the spacing prior.
    spacing_weight : float
        Relative weight of the spacing prior vs data fidelity.
    anchor_weight : float
        Multiplier for anchor data-fidelity terms (default 5x).
    """
    import numpy as np
    from scipy.optimize import minimize

    raw = np.array([s.position_mm for s in slices], dtype=np.float64)
    n = len(raw)

    if n <= 1:
        return list(slices)

    # Per-slice weights: anchors are more reliable (two-stage estimation)
    weights = np.array(
        [anchor_weight if s.source == "anchor" else 1.0 for s in slices],
        dtype=np.float64,
    )

    def _huber(residuals: np.ndarray, delta: float) -> np.ndarray:
        """Pseudo-Huber loss: smooth approximation, differentiable everywhere."""
        return delta**2 * (np.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)

    def objective(x: np.ndarray) -> float:
        # Data fidelity: keep fitted positions close to VLM estimates
        # Anchors are weighted higher to anchor the fit at reliable points
        data_loss = float(np.sum(weights * _huber(x - raw, data_delta_mm)))

        # Spacing prior: penalize local intervals deviating from expected
        intervals = np.diff(x)
        spacing_loss = float(
            np.sum(_huber(intervals - interval_mm, interval_delta_mm))
        )

        return data_loss + spacing_weight * spacing_loss

    # Hard constraints: x[i] >= x[i-1] + thickness_mm
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x, i=i: x[i] - x[i - 1] - thickness_mm,
        }
        for i in range(1, n)
    ]

    # Initial guess: sort raw estimates and enforce minimum spacing
    x0 = np.sort(raw)
    for i in range(1, n):
        if x0[i] < x0[i - 1] + thickness_mm:
            x0[i] = x0[i - 1] + thickness_mm

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        logger.warning(
            "Isotonic optimizer did not converge: %s (nit=%d). "
            "Using best iterate.",
            result.message,
            result.nit,
        )
    final = result.x

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
