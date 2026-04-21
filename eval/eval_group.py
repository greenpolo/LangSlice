"""Eval harness for multi-slice group AP estimation.

Runs ``estimate_group()`` on consecutive-slice groups formed from a golden
dataset, compares estimated positions to ground truth, and outputs structured
JSON metrics to stdout.

Group formation rules:
  - Slices are ordered by ``slice_index`` from the ground truth file.
  - Continuous runs are determined by detecting AP-position gaps larger than
    ``--gap-threshold`` mm (default 0.5 mm).  Gaps indicate wellplate
    boundaries or missing slices and are never bridged by a group.
  - Within each continuous run, groups of ``--group-size`` (default 4)
    consecutive slices are formed.  A leftover tail of ≥2 slices is included
    as a smaller group; a tail of 1 slice is appended to the preceding group
    (making it size+1) or dropped if the run had only one slice to begin with.

Usage:
    python eval/eval_group.py \\
        --images references/TestImages/M01 \\
        --ground-truth references/TestImages/M01/ground_truth.json \\
        [--group-size 4] \\
        [--gap-threshold 0.5] \\
        [--model MODEL_NAME] \\
        [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Repo root on sys.path so ``langslice`` resolves when run directly.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# slice-bench for shared metrics utilities
_SLICE_BENCH = os.path.join(_REPO_ROOT, "slice-bench")
if _SLICE_BENCH not in sys.path:
    sys.path.insert(0, _SLICE_BENCH)

logger = logging.getLogger(__name__)

# Hard-coded per spec; thin slice thickness is not expected to vary in eval.
_THICKNESS_UM = 50

# Fallback indicator phrase written by estimate_group() when the model never
# called submit_group_estimate within the iteration budget.
_FALLBACK_PHRASE = "Model did not submit"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multi-slice group AP estimation against ground truth"
    )
    p.add_argument(
        "--images",
        required=True,
        help="Directory containing slice images",
    )
    p.add_argument(
        "--ground-truth",
        required=True,
        help="Ground truth JSON file",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Target number of slices per group (default: 4)",
    )
    p.add_argument(
        "--gap-threshold",
        type=float,
        default=0.5,
        help=(
            "AP gap in mm that marks a discontinuity between consecutive "
            "slices (default: 0.5)"
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the Gemini model name passed to estimate_group()",
    )
    p.add_argument(
        "--media-resolution",
        default=None,
        choices=["low", "medium", "high", "ultra_high"],
        help="Gemini media resolution for input images (default: auto per model)",
    )
    p.add_argument(
        "--thinking",
        default=None,
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        help="Override Gemini thinking level",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Max tool-loop iterations per group (default: 25)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON to stdout",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Run all groups concurrently using threads",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="Send atlas slices as a composite grid instead of individually",
    )
    p.add_argument(
        "--provider",
        default="google",
        choices=["google", "openai"],
        help="VLM provider to use (default: google)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------


def _load_ground_truth(path: str) -> dict[str, dict]:
    """Return the raw ground truth dict keyed by filename."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ordered_slices(gt: dict[str, dict]) -> list[tuple[str, float]]:
    """Return [(filename, ap_mm), ...] sorted by slice_index."""
    items = sorted(gt.items(), key=lambda kv: kv[1].get("slice_index", 0))
    return [(fname, entry["ap_mm"]) for fname, entry in items]


# ---------------------------------------------------------------------------
# Group formation
# ---------------------------------------------------------------------------


def _split_into_runs(
    ordered: list[tuple[str, float]],
    gap_threshold_mm: float,
) -> list[list[tuple[str, float]]]:
    """Split an ordered slice list into continuous runs.

    A new run starts whenever the AP difference between two consecutive slices
    exceeds ``gap_threshold_mm``.
    """
    if not ordered:
        return []

    runs: list[list[tuple[str, float]]] = []
    current_run: list[tuple[str, float]] = [ordered[0]]

    for fname, ap_mm in ordered[1:]:
        prev_ap = current_run[-1][1]
        if abs(ap_mm - prev_ap) > gap_threshold_mm:
            runs.append(current_run)
            current_run = [(fname, ap_mm)]
        else:
            current_run.append((fname, ap_mm))

    runs.append(current_run)
    return runs


def _form_groups(
    run: list[tuple[str, float]],
    group_size: int,
    max_group_size: int = 8,
) -> list[list[tuple[str, float]]]:
    """Partition a continuous run into groups of approximately ``group_size``.

    Rules:
      - Full groups of exactly ``group_size`` are formed first.
      - A leftover tail of ≥2 is emitted as its own smaller group.
      - A leftover tail of exactly 1 is absorbed into the last group
        (making it size+1) rather than forming an invalid single-slice group,
        unless that would exceed ``max_group_size``.
      - If the entire run is 1 slice it is dropped (cannot form a group).
    """
    n = len(run)
    if n < 2:
        return []

    groups: list[list[tuple[str, float]]] = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= group_size:
            chunk = run[start:]
            if len(chunk) == 1 and groups and len(groups[-1]) < max_group_size:
                # Absorb the lone tail into the previous group
                groups[-1] = groups[-1] + chunk
            elif len(chunk) >= 2:
                groups.append(chunk)
            # else: single slice with no prior group — silently drop
            break
        groups.append(run[start : start + group_size])
        start += group_size

    return groups


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_and_prep(image_dir: str, filename: str):
    """Load a slice image and apply legacy eval-side VLM preprocessing."""
    from PIL import Image

    from langslice.image_prep import adaptive_preprocess, normalize_image, prepare_image_for_vlm

    path = os.path.join(image_dir, filename)
    raw = Image.open(path)
    canonical = normalize_image(raw)
    prep = prepare_image_for_vlm(canonical)
    img = adaptive_preprocess(prep.image)
    return img


def _load_raw_image(image_dir: str, filename: str):
    """Load a slice image; Google/ADK preprocessing happens inside the runner."""
    from PIL import Image

    path = os.path.join(image_dir, filename)
    with Image.open(path) as raw:
        return raw.copy()


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------


def _run_eval(args: argparse.Namespace) -> dict:
    from slice_bench.metrics import compute_metrics

    if args.provider == "openai":
        from langslice.estimation.openai.ap_multi_slice import estimate_group
    else:
        import langslice.vlm_config as vlm_config
        from langslice.estimation import estimate_group

        # Apply thinking level override before any API calls
        if args.thinking:
            vlm_config.set_thinking_level(args.thinking)

    gt_raw = _load_ground_truth(args.ground_truth)
    ordered = _ordered_slices(gt_raw)

    atlas_name: str = next(iter(gt_raw.values()))["atlas"]

    runs = _split_into_runs(ordered, args.gap_threshold)
    logger.info(
        "Found %d continuous run(s) from %d slices (gap threshold %.2f mm)",
        len(runs),
        len(ordered),
        args.gap_threshold,
    )

    # Assign each run's slices to groups
    all_groups: list[list[tuple[str, float]]] = []
    for run in runs:
        all_groups.extend(_form_groups(run, args.group_size))

    logger.info("Formed %d group(s) across all runs", len(all_groups))

    # ---- Per-group results ----
    estimates: dict[str, float] = {}
    ground_truth_flat: dict[str, float] = {
        fname: ap_mm for fname, ap_mm in ordered
    }

    group_records: list[dict] = []
    per_slice_records: list[dict] = []
    n_failures = 0
    n_fallbacks = 0
    total_wall_time_s = 0.0

    def _run_single_group(
        group_idx: int, group: list[tuple[str, float]]
    ) -> tuple[dict, dict[str, float]]:
        """Run estimation for one group, return (record, estimates_dict)."""
        filenames = [fname for fname, _ in group]
        gt_positions = [ap_mm for _, ap_mm in group]

        if len(gt_positions) >= 2:
            gaps = [
                abs(gt_positions[i + 1] - gt_positions[i])
                for i in range(len(gt_positions) - 1)
            ]
            avg_interval_mm = sum(gaps) / len(gaps)
        else:
            avg_interval_mm = 0.200
        interval_um = round(avg_interval_mm * 1000)

        def _progress(msg: str, gidx: int = group_idx) -> None:
            print(f"[eval] group {gidx + 1}: {msg}", file=sys.stderr)

        _progress(f"slices={filenames}, interval={interval_um}µm")

        try:
            if args.provider == "openai":
                images = [_load_and_prep(args.images, fn) for fn in filenames]
            else:
                images = [_load_raw_image(args.images, fn) for fn in filenames]
        except Exception as exc:
            logger.error("Group %d: image load failed: %s", group_idx, exc)
            return {
                "group_index": group_idx,
                "slices": filenames,
                "interval_um": interval_um,
                "wall_time_s": 0.0,
                "status": "image_load_failure",
                "error": str(exc),
            }, {}

        t0 = time.time()
        try:
            estimate_kwargs: dict = dict(
                images=images,
                atlas_name=atlas_name,
                interval_um=interval_um,
                thickness_um=_THICKNESS_UM,
                model_name=args.model,
                max_iterations=args.max_iterations,
                send_individually=not args.grid,
                on_progress=_progress,
            )
            if args.provider != "openai":
                estimate_kwargs["media_resolution"] = args.media_resolution
            result = estimate_group(**estimate_kwargs)
            wall_time_s = round(time.time() - t0, 2)

            is_fallback = _FALLBACK_PHRASE in result.group_reasoning

            if is_fallback:
                _progress(f"fallback triggered after {wall_time_s:.1f}s")
            else:
                _progress(
                    f"completed in {wall_time_s:.1f}s — "
                    + ", ".join(f"{p.position_mm:.3f}" for p in result.positions)
                    + " mm"
                )

            group_estimates: dict[str, float] = {}
            for i, (fname, _) in enumerate(group):
                if i < len(result.positions):
                    group_estimates[fname] = result.positions[i].position_mm

            return {
                "group_index": group_idx,
                "slices": filenames,
                "interval_um": interval_um,
                "wall_time_s": wall_time_s,
                "fallback": is_fallback,
            }, group_estimates

        except Exception as exc:
            wall_time_s = round(time.time() - t0, 2)
            logger.error("Group %d: estimate_group() failed: %s", group_idx, exc)
            return {
                "group_index": group_idx,
                "slices": filenames,
                "interval_um": interval_um,
                "wall_time_s": wall_time_s,
                "status": "estimation_failure",
                "error": str(exc),
            }, {}

    if args.parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=len(all_groups)) as executor:
            for gidx, group in enumerate(all_groups):
                fut = executor.submit(_run_single_group, gidx, group)
                futures[fut] = gidx
            for fut in as_completed(futures):
                record, group_est = fut.result()
                group_records.append(record)
                estimates.update(group_est)
                total_wall_time_s += record.get("wall_time_s", 0.0)
                if record.get("status") in ("image_load_failure", "estimation_failure"):
                    n_failures += 1
                if record.get("fallback"):
                    n_fallbacks += 1
        group_records.sort(key=lambda r: r["group_index"])
    else:
        for group_idx, group in enumerate(all_groups):
            record, group_est = _run_single_group(group_idx, group)
            group_records.append(record)
            estimates.update(group_est)
            total_wall_time_s += record.get("wall_time_s", 0.0)
            if record.get("status") in ("image_load_failure", "estimation_failure"):
                n_failures += 1
            if record.get("fallback"):
                n_fallbacks += 1

    # ---- Metrics ----
    bench = compute_metrics(
        model_name=args.model or "default",
        dataset=os.path.basename(os.path.dirname(args.ground_truth)),
        estimates=estimates,
        ground_truth=ground_truth_flat,
    )

    # Build per_slice records with group membership
    fname_to_group: dict[str, int] = {}
    for rec in group_records:
        for fn in rec["slices"]:
            fname_to_group[fn] = rec["group_index"]

    for entry in bench.per_slice:
        fname = entry["filename"]
        entry["group_index"] = fname_to_group.get(fname, -1)
        per_slice_records.append(entry)

    n_groups_attempted = len(all_groups)
    mean_wall = (
        round(total_wall_time_s / n_groups_attempted, 2)
        if n_groups_attempted > 0
        else 0.0
    )

    return {
        "summary": {
            "mae_mm": bench.mae_mm,
            "median_error_mm": bench.median_error_mm,
            "max_error_mm": bench.max_error_mm,
            "pct_within_0.25mm": bench.pct_within_025,
            "pct_within_0.5mm": bench.pct_within_050,
            "pct_within_1.0mm": bench.pct_within_100,
            "n_slices": bench.n_slices,
            "n_groups": n_groups_attempted,
            "n_failures": n_failures,
            "n_fallbacks": n_fallbacks,
            "total_wall_time_s": round(total_wall_time_s, 2),
            "mean_wall_time_per_group_s": mean_wall,
        },
        "groups": group_records,
        "per_slice": per_slice_records,
        "config": {
            "provider": args.provider,
            "model": args.model,
            "group_size": args.group_size,
            "gap_threshold": args.gap_threshold,
            "media_resolution": args.media_resolution,
            "send_individually": not args.grid,
            "thinking": args.thinking,
            "max_iterations": args.max_iterations,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Suppress debug artifacts during eval runs
    os.environ.pop("LANGSLICE_VLM_DEBUG_DIR", None)

    result = _run_eval(args)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        s = result["summary"]
        print(f"MAE:                    {s['mae_mm']:.4f} mm")
        print(f"Median error:           {s['median_error_mm']:.4f} mm")
        print(f"Max error:              {s['max_error_mm']:.4f} mm")
        print(f"Within 0.25mm:          {s['pct_within_0.25mm']:.1%}")
        print(f"Within 0.50mm:          {s['pct_within_0.5mm']:.1%}")
        print(f"Within 1.00mm:          {s['pct_within_1.0mm']:.1%}")
        print(f"Slices evaluated:       {s['n_slices']}")
        print(f"Groups attempted:       {s['n_groups']}")
        print(f"Hard failures:          {s['n_failures']}")
        print(f"Fallbacks (no submit):  {s['n_fallbacks']}")
        print(f"Total wall time:        {s['total_wall_time_s']:.1f}s")
        print(f"Mean wall time/group:   {s['mean_wall_time_per_group_s']:.1f}s")

        # Show worst per-slice errors
        errors = [
            e for e in result["per_slice"]
            if e.get("error_mm") is not None
        ]
        if errors:
            worst = sorted(errors, key=lambda x: x["error_mm"], reverse=True)[:5]
            print(f"\nWorst errors (top {len(worst)}):")
            for e in worst:
                print(
                    f"  {e['filename']}: "
                    f"est={e.get('estimated_mm', '?'):.3f} "
                    f"gt={e['ground_truth_mm']:.3f} "
                    f"err={e['error_mm']:.3f}mm "
                    f"(group {e['group_index']})"
                )


if __name__ == "__main__":
    main()
