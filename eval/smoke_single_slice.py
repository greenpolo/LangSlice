"""Phase 3 smoke test: single-slice ADK estimator on 5 mid-brain M01 slices.

Invokes the new langslice.harness.estimation.estimate_position (ADK-backed)
against 5 evenly-spaced mid-brain slices from references/TestImages/M01.
Prints per-slice errors and an overall PASS/FAIL verdict against the Task 3.8
gate: no errors, no fallbacks, all errors within 2 mm of ground truth.

Not committed to main; dev-only harness validation script.
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

# google-genai checks GOOGLE_API_KEY first, then GEMINI_API_KEY. Make sure ADK
# sees whichever is set.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from PIL import Image  # noqa: E402

from langslice.harness.estimation import estimate_position  # noqa: E402

TEST_IMAGES_DIR = Path(r"C:\LabSoftware\LangSlice\references\TestImages\M01")
GT_PATH = TEST_IMAGES_DIR / "ground_truth.json"
MODEL = "gemini-3-flash-preview"
ATLAS = "allen_mouse_25um"

# 5 evenly-spaced mid-brain slices (AP 4.0-7.0mm in M01)
SLICES = [
    "M01_001_008.tif",  # 4.052 mm
    "M01_001_011.tif",  # 4.532 mm
    "M01_002_005.tif",  # 5.620 mm
    "M01_002_008.tif",  # 6.133 mm
    "M01_003_001.tif",  # 6.616 mm
]


INTER_SLICE_DELAY_S = 20  # pace to stay under 2M input-tokens-per-minute quota


def main() -> int:
    gt = json.loads(GT_PATH.read_text())
    results: list[dict] = []
    fallback_phrase = "Model did not submit within iteration+retry budget"

    for i, name in enumerate(SLICES):
        if i > 0:
            print(f"(pacing: sleeping {INTER_SLICE_DELAY_S}s)", flush=True)
            time.sleep(INTER_SLICE_DELAY_S)
        gt_ap = float(gt[name]["ap_mm"])
        img_path = TEST_IMAGES_DIR / name
        print(f"\n=== {name} (GT={gt_ap:.3f}mm) ===", flush=True)
        t0 = time.time()
        try:
            img = Image.open(img_path)
            result = estimate_position(
                img, ATLAS, plane="coronal", model=MODEL, max_iterations=20,
            )
            elapsed = time.time() - t0
            error = abs(result.position_mm - gt_ap)
            is_fallback = fallback_phrase in result.reasoning
            print(
                f"  predicted: {result.position_mm:.3f}mm  error: {error:.3f}mm  "
                f"{'FALLBACK' if is_fallback else ''}  ({elapsed:.1f}s)"
            )
            results.append({
                "name": name,
                "gt_mm": gt_ap,
                "predicted_mm": result.position_mm,
                "error_mm": error,
                "fallback": is_fallback,
                "elapsed_s": elapsed,
                "reasoning": result.reasoning[:160],
            })
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR: {type(exc).__name__}: {exc}  ({elapsed:.1f}s)")
            results.append({
                "name": name,
                "gt_mm": gt_ap,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": elapsed,
            })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_errors = sum(1 for r in results if "error" in r)
    n_fallbacks = sum(1 for r in results if r.get("fallback"))
    errors_mm = [r["error_mm"] for r in results if "error_mm" in r]
    worst = max(errors_mm) if errors_mm else float("nan")
    mae = sum(errors_mm) / len(errors_mm) if errors_mm else float("nan")
    print(f"Slices run: {len(results)}")
    print(f"Exceptions: {n_errors}")
    print(f"Fallbacks: {n_fallbacks}")
    print(f"MAE: {mae:.3f}mm")
    print(f"Worst error: {worst:.3f}mm")

    # Gate check
    gate_pass = n_errors == 0 and n_fallbacks == 0 and worst < 2.0
    print(f"\nGATE: {'PASS' if gate_pass else 'FAIL'}")

    # Dump JSON to eval_outputs (gitignored)
    out_dir = REPO_ROOT / "eval_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "post_step3_single_M01.json"
    summary = {
        "task": "3.8 single-slice smoke",
        "model": MODEL,
        "atlas": ATLAS,
        "slices": results,
        "summary": {
            "n_slices": len(results),
            "n_errors": n_errors,
            "n_fallbacks": n_fallbacks,
            "mae_mm": mae,
            "worst_error_mm": worst,
            "gate_pass": gate_pass,
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
