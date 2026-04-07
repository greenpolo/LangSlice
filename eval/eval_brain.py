"""Eval harness for whole-brain AP estimation.

Runs the brain pipeline on a golden dataset, compares estimated positions
to ground truth, and outputs structured JSON metrics to stdout.

Usage:
    python eval/eval_brain.py \
        --images references/TestImages/M01 \
        --ground-truth references/TestImages/M01/ground_truth.json \
        --coarse-model gemma-4-31b-it \
        --fine-model gemma-4-31b-it \
        --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the repo root is importable so ``langslice`` resolves even when the
# script is run directly via ``python eval/eval_brain.py``.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)

FAIL_THRESHOLD_MM = 0.1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate brain AP estimation against ground truth")
    p.add_argument("--images", required=True, help="Directory containing slice images")
    p.add_argument("--ground-truth", required=True, help="Ground truth JSON file")
    p.add_argument("--coarse-model", required=True, help="Model for anchor estimation")
    p.add_argument("--fine-model", required=True, help="Model for non-anchor estimation")
    p.add_argument("--json", action="store_true", help="Output structured JSON to stdout")
    p.add_argument("--threshold", type=float, default=FAIL_THRESHOLD_MM, help="Failure threshold in mm")
    p.add_argument("--n-anchors", type=int, default=4, help="Number of anchor slices")
    return p.parse_args()


def _load_ground_truth(path: str) -> dict[str, float]:
    """Load ground truth JSON and return {filename: ap_mm} mapping."""
    with open(path) as f:
        data = json.load(f)
    return {filename: entry["ap_mm"] for filename, entry in data.items()}


def _build_config(
    image_folder: str,
    atlas_name: str,
    coarse_model: str,
    fine_model: str,
    n_anchors: int = 4,
) -> "BrainEstimationConfig":
    """Build a BrainEstimationConfig using the pipeline's current defaults.

    The agent may have changed defaults in the code (n_anchors, ordering, etc.).
    We import the config class at call time so we pick up those changes.
    """
    from langslice.brain.types import BrainEstimationConfig

    return BrainEstimationConfig(
        image_folder=image_folder,
        atlas_name=atlas_name,
        thickness_um=50,
        interval_um=200,
        n_anchors=n_anchors,
        ordering="strict",
        refinement=True,
        max_parallel=10,
        z_axis="AP",
        coarse_model=coarse_model,
        fine_model=fine_model,
    )


def _compute_metrics(
    ground_truth: dict[str, float],
    slices: list,
    threshold_mm: float,
) -> dict:
    """Compare estimated positions to ground truth and compute metrics."""
    per_slice = []
    errors = []

    for s in slices:
        gt_mm = ground_truth.get(s.filename)
        if gt_mm is None:
            continue
        error_mm = abs(s.position_mm - gt_mm)
        errors.append(error_mm)
        per_slice.append({
            "filename": s.filename,
            "estimated_mm": round(s.position_mm, 4),
            "ground_truth_mm": gt_mm,
            "error_mm": round(error_mm, 4),
            "source": s.source,
            "status": "pass" if error_mm <= threshold_mm else "fail",
        })

    n = len(errors)
    if n == 0:
        return {"summary": {"error": "No slices matched ground truth"}, "per_slice": []}

    n_failing = sum(1 for e in errors if e > threshold_mm)
    n_passing = n - n_failing
    pct_within_01 = n_passing / n

    return {
        "summary": {
            "mae_mm": round(statistics.mean(errors), 4),
            "median_error_mm": round(statistics.median(errors), 4),
            "max_error_mm": round(max(errors), 4),
            "pct_within_0.1mm": round(pct_within_01, 4),
            "pct_within_0.25mm": round(sum(1 for e in errors if e <= 0.25) / n, 4),
            "pct_within_0.5mm": round(sum(1 for e in errors if e <= 0.5) / n, 4),
            "n_slices": n,
            "n_failing": n_failing,
            "n_passing": n_passing,
            "failing_threshold_mm": threshold_mm,
            "accuracy": round(pct_within_01, 4),
        },
        "per_slice": per_slice,
    }


async def _run(args: argparse.Namespace) -> dict:
    from langslice.brain.pipeline import run_brain_estimation

    ground_truth = _load_ground_truth(args.ground_truth)

    # Derive atlas name from ground truth (all entries use the same atlas)
    gt_entries = list(json.load(open(args.ground_truth)).values())
    atlas_name = gt_entries[0]["atlas"] if gt_entries else "allen_mouse_25um"

    config = _build_config(
        image_folder=args.images,
        atlas_name=atlas_name,
        coarse_model=args.coarse_model,
        fine_model=args.fine_model,
        n_anchors=args.n_anchors,
    )

    # Disable debug artifact dumps during eval runs to avoid filling disk.
    os.environ.pop("LANGSLICE_VLM_DEBUG_DIR", None)

    def _progress(msg: str) -> None:
        print(f"[eval] {msg}", file=sys.stderr)

    # Use a temp checkpoint path so we never load stale checkpoints from the
    # golden data directory.  The file is cleaned up after the run.
    import tempfile
    cp_fd, cp_path = tempfile.mkstemp(suffix=".json", prefix="eval_brain_cp_")
    os.close(cp_fd)
    os.unlink(cp_path)  # pipeline creates it fresh; we just need a clean path

    t0 = time.time()
    result = await run_brain_estimation(
        config, checkpoint_path=cp_path, on_progress=_progress
    )
    elapsed_s = time.time() - t0

    # Clean up temp checkpoint
    if os.path.exists(cp_path):
        os.unlink(cp_path)

    metrics = _compute_metrics(ground_truth, result.slices, args.threshold)
    metrics["config"] = {
        "coarse_model": args.coarse_model,
        "fine_model": args.fine_model,
        "n_anchors": config.n_anchors,
        "ordering": config.ordering,
        "refinement": config.refinement,
        "thickness_um": config.thickness_um,
        "interval_um": config.interval_um,
    }
    metrics["summary"]["elapsed_s"] = round(elapsed_s, 1)

    return metrics


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    metrics = asyncio.run(_run(args))

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        s = metrics["summary"]
        print(f"MAE:              {s['mae_mm']:.4f} mm")
        print(f"Median error:     {s['median_error_mm']:.4f} mm")
        print(f"Max error:        {s['max_error_mm']:.4f} mm")
        print(f"Within 0.1mm:     {s['pct_within_0.1mm']:.1%}")
        print(f"Within 0.25mm:    {s['pct_within_0.25mm']:.1%}")
        print(f"Within 0.5mm:     {s['pct_within_0.5mm']:.1%}")
        print(f"Passing/Failing:  {s['n_passing']}/{s['n_failing']} (of {s['n_slices']})")
        print(f"Elapsed:          {s['elapsed_s']:.1f}s")

        failing = [p for p in metrics["per_slice"] if p["status"] == "fail"]
        if failing:
            print(f"\nFailing cases ({len(failing)}):")
            for p in sorted(failing, key=lambda x: x["error_mm"], reverse=True):
                print(
                    f"  {p['filename']}: "
                    f"est={p['estimated_mm']:.3f} gt={p['ground_truth_mm']:.3f} "
                    f"err={p['error_mm']:.3f}mm ({p['source']})"
                )


if __name__ == "__main__":
    main()
