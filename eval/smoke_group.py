"""Phase 4 Task 4.4 smoke test: multi-slice group ADK estimator on M01.

Invokes the new langslice.harness.estimation.estimate_group (ADK-backed) on a
single group of 4 consecutive mid-brain M01 slices via Flash. Prints per-slice
errors and an overall PASS/FAIL verdict against the Task 4.4 gate: no exception,
no fallback, all errors within 2 mm of ground truth.

Parallels eval/smoke_single_slice.py. Not committed output — JSON written to
eval_outputs/ (gitignored). Dev-only harness validation script.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set.", file=sys.stderr)
    sys.exit(2)

# google-genai / ADK checks GOOGLE_API_KEY first, then GEMINI_API_KEY. Make sure
# ADK sees whichever is set.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from PIL import Image  # noqa: E402

from langslice.harness.estimation import estimate_group  # noqa: E402

TEST_IMAGES_DIR = Path(r"C:\LabSoftware\LangSlice\references\TestImages\M01")
GT_PATH = TEST_IMAGES_DIR / "ground_truth.json"
MODEL = "gemini-3-flash-preview"
ATLAS = "allen_mouse_25um"

# 4 consecutive mid-brain slices from M01 wellplate 2.
# slice_index values are 13, 14, 15, 16 (contiguous, same wellplate run).
# AP range 5.620 - 6.133 mm — solidly mid-brain.
SLICES = [
    "M01_002_005.tif",  # 5.620 mm  (slice_index 13)
    "M01_002_006.tif",  # 5.734 mm  (slice_index 14)
    "M01_002_007.tif",  # 5.931 mm  (slice_index 15)
    "M01_002_008.tif",  # 6.133 mm  (slice_index 16)
]


def main() -> int:
    gt = json.loads(GT_PATH.read_text())
    fallback_phrase = "Model did not submit"

    # Build GT position list and compute average interval.
    gt_positions: list[float] = [float(gt[name]["ap_mm"]) for name in SLICES]
    gaps = [
        abs(gt_positions[i + 1] - gt_positions[i])
        for i in range(len(gt_positions) - 1)
    ]
    avg_interval_mm = sum(gaps) / len(gaps)
    interval_um = round(avg_interval_mm * 1000)

    print(f"Group: {len(SLICES)} slices from M01 wellplate 2 (mid-brain)", flush=True)
    for name, gt_ap in zip(SLICES, gt_positions, strict=False):
        print(f"  {name}: GT={gt_ap:.3f} mm (slice_index={gt[name]['slice_index']})")
    print(f"Average interval: {avg_interval_mm:.3f} mm ({interval_um} um)", flush=True)
    print(f"Model: {MODEL}  Atlas: {ATLAS}", flush=True)

    # Load raw PIL images. run_group_session applies CLAHE internally
    # (apply_clahe=True default); the shim does not override it. Do NOT
    # pre-CLAHE here or it would be double-applied.
    images: list[Image.Image] = []
    for name in SLICES:
        path = TEST_IMAGES_DIR / name
        images.append(Image.open(path))

    print("\nRunning estimate_group ...", flush=True)
    t0 = time.time()
    error_payload: dict | None = None
    result = None
    try:
        result = estimate_group(
            images=images,
            atlas_name=ATLAS,
            interval_um=interval_um,
            thickness_um=50,
            model_name=MODEL,
            max_iterations=25,
        )
    except Exception as exc:  # noqa: BLE001 — smoke test captures everything
        elapsed = time.time() - t0
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "elapsed_s": elapsed,
        }
        print(f"  ERROR: {error_payload['type']}: {error_payload['message']}  ({elapsed:.1f}s)")
    else:
        elapsed = time.time() - t0
        print(f"  completed in {elapsed:.1f}s", flush=True)

    # ---- Build per-slice records ----
    slice_records: list[dict] = []
    if result is not None:
        for name, gt_ap, pos in zip(SLICES, gt_positions, result.positions, strict=True):
            predicted = float(pos.position_mm)
            err = abs(predicted - gt_ap)
            slice_records.append({
                "name": name,
                "gt_mm": gt_ap,
                "predicted_mm": predicted,
                "error_mm": err,
                "slice_index": gt[name]["slice_index"],
            })
    else:
        for name, gt_ap in zip(SLICES, gt_positions, strict=True):
            slice_records.append({
                "name": name,
                "gt_mm": gt_ap,
                "slice_index": gt[name]["slice_index"],
            })

    # ---- Summary / gate ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if error_payload is not None:
        print(f"Exception: {error_payload['type']}: {error_payload['message']}")
        fallback = False
        errors_mm: list[float] = []
        mae = float("nan")
        max_err = float("nan")
        reasoning_str = ""
    else:
        assert result is not None  # for type checker
        reasoning_str = result.group_reasoning
        fallback = fallback_phrase in reasoning_str
        errors_mm = [r["error_mm"] for r in slice_records]
        mae = sum(errors_mm) / len(errors_mm)
        max_err = max(errors_mm)
        for r in slice_records:
            print(
                f"  {r['name']}: pred={r['predicted_mm']:.3f} "
                f"gt={r['gt_mm']:.3f} err={r['error_mm']:.3f} mm"
            )
        print(f"\nMAE:         {mae:.3f} mm")
        print(f"Max error:   {max_err:.3f} mm")
        print(f"Fallback:    {fallback}")
        print(f"Wall time:   {elapsed:.1f} s")

    gate_pass = (
        error_payload is None
        and not fallback
        and len(errors_mm) == len(SLICES)
        and max_err < 2.0
    )
    print(f"\nGATE: {'PASS' if gate_pass else 'FAIL'}")
    if gate_pass and mae > 1.0:
        print("CONCERN: MAE > 1 mm — gate passed but accuracy is worse than expected.")

    # ---- Dump JSON ----
    out_dir = REPO_ROOT / "eval_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "post_step4_group_M01.json"
    payload: dict = {
        "task": "4.4 group smoke",
        "model": MODEL,
        "atlas": ATLAS,
        "interval_um": interval_um,
        "slices": slice_records,
        "summary": {
            "n_slices": len(SLICES),
            "fallback": fallback,
            "mae_mm": mae,
            "max_error_mm": max_err,
            "wall_time_s": elapsed,
            "gate_pass": gate_pass,
        },
        "reasoning": reasoning_str,
    }
    if error_payload is not None:
        payload["error"] = error_payload

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
