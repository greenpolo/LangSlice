"""Benchmark metrics for AP estimation evaluation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a single model run."""

    model_name: str
    dataset: str
    n_slices: int
    mae_mm: float
    median_error_mm: float
    max_error_mm: float
    pct_within_025: float  # % of slices within 0.25mm
    pct_within_050: float  # % of slices within 0.50mm
    pct_within_100: float  # % of slices within 1.00mm
    per_slice: list[dict]


def compute_metrics(
    model_name: str,
    dataset: str,
    estimates: dict[str, float],
    ground_truth: dict[str, float],
) -> BenchmarkResult:
    """Compute benchmark metrics from estimates vs ground truth.

    Args:
        model_name: Identifier for the model being evaluated.
        dataset: Name/path of the dataset used.
        estimates: {filename: estimated_ap_mm} from the model.
        ground_truth: {filename: true_ap_mm} from annotations.

    Returns:
        BenchmarkResult with all metrics computed.
    """
    per_slice = []
    errors = []

    for filename, true_mm in ground_truth.items():
        est_mm = estimates.get(filename)
        if est_mm is None:
            per_slice.append(
                {"filename": filename, "ground_truth_mm": true_mm, "error_mm": None}
            )
            continue
        error = abs(est_mm - true_mm)
        errors.append(error)
        per_slice.append({
            "filename": filename,
            "estimated_mm": round(est_mm, 4),
            "ground_truth_mm": true_mm,
            "error_mm": round(error, 4),
        })

    n = len(errors)
    if n == 0:
        return BenchmarkResult(
            model_name=model_name,
            dataset=dataset,
            n_slices=0,
            mae_mm=float("inf"),
            median_error_mm=float("inf"),
            max_error_mm=float("inf"),
            pct_within_025=0.0,
            pct_within_050=0.0,
            pct_within_100=0.0,
            per_slice=per_slice,
        )

    return BenchmarkResult(
        model_name=model_name,
        dataset=dataset,
        n_slices=n,
        mae_mm=round(statistics.mean(errors), 4),
        median_error_mm=round(statistics.median(errors), 4),
        max_error_mm=round(max(errors), 4),
        pct_within_025=round(sum(1 for e in errors if e <= 0.25) / n, 4),
        pct_within_050=round(sum(1 for e in errors if e <= 0.50) / n, 4),
        pct_within_100=round(sum(1 for e in errors if e <= 1.00) / n, 4),
        per_slice=per_slice,
    )
