"""CLI entry point for slice-bench evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slice_bench.metrics import compute_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="slice-bench: evaluate AP estimation models"
    )
    p.add_argument(
        "--adapter",
        required=True,
        choices=["gemini", "gemma-local"],
        help="Model adapter to evaluate",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Directory containing slice images",
    )
    p.add_argument(
        "--ground-truth",
        required=True,
        help="Ground truth JSON file ({filename: {ap_mm: float}})",
    )
    p.add_argument(
        "--atlas",
        default="allen_mouse_25um",
        help="BrainGlobe atlas name",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to local model weights (for gemma-local adapter)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: stdout)",
    )
    return p.parse_args()


def _load_ground_truth(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Support both {filename: ap_mm} and {filename: {ap_mm: float}} formats
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = v["ap_mm"]
        else:
            result[k] = float(v)
    return result


def _get_adapter(args: argparse.Namespace):
    if args.adapter == "gemini":
        from slice_bench.adapters.gemini import GeminiAdapter

        return GeminiAdapter()
    elif args.adapter == "gemma-local":
        from slice_bench.adapters.gemma_local import GemmaLocalAdapter

        if not args.model_path:
            print("ERROR: --model-path required for gemma-local adapter", file=sys.stderr)
            sys.exit(1)
        return GemmaLocalAdapter(model_path=args.model_path)
    else:
        print(f"ERROR: Unknown adapter: {args.adapter}", file=sys.stderr)
        sys.exit(1)


async def _main():
    args = _parse_args()

    dataset_dir = Path(args.dataset)
    images = sorted(
        p for p in dataset_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".tif", ".tiff"}
    )

    if not images:
        print(f"ERROR: No images found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    ground_truth = _load_ground_truth(args.ground_truth)
    adapter = _get_adapter(args)

    print(f"Running {adapter.name} on {len(images)} images...", file=sys.stderr)
    estimates = await adapter.estimate_positions(images, args.atlas)

    result = compute_metrics(
        model_name=adapter.name,
        dataset=args.dataset,
        estimates=estimates,
        ground_truth=ground_truth,
    )

    output = {
        "model": result.model_name,
        "dataset": result.dataset,
        "n_slices": result.n_slices,
        "mae_mm": result.mae_mm,
        "median_error_mm": result.median_error_mm,
        "max_error_mm": result.max_error_mm,
        "pct_within_0.25mm": result.pct_within_025,
        "pct_within_0.50mm": result.pct_within_050,
        "pct_within_1.00mm": result.pct_within_100,
        "per_slice": result.per_slice,
    }

    json_str = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    asyncio.run(_main())
