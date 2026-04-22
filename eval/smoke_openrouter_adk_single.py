"""Run a staged single-slice ADK eval through any supported ADK model string.

Default behavior runs 12 evenly-spaced slices from the ground-truth file.
For LiteLLM/OpenRouter/Ollama models, use this only after
``eval/probe_openrouter_transport.py`` passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

from langslice.harness.estimation.runner import run_single_slice_session  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

FALLBACK_PHRASE = "Model did not submit within iteration+retry budget"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        default=str(REPO_ROOT.parent.parent / "references" / "TestImages" / "M01"),
        help="Directory containing slice images",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Ground truth JSON path; defaults to <images>/ground_truth.json",
    )
    parser.add_argument("--atlas", default="allen_mouse_25um")
    parser.add_argument("--model", default="litellm-proxy:langslice-qwen36-plus")
    parser.add_argument("--media-resolution", default="medium")
    parser.add_argument(
        "--multimodal-history",
        default="persistent",
        choices=["persistent", "one_turn"],
        help="ADK multimodal image replay mode: persistent=new, one_turn=old stock behavior",
    )
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--model-call-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep before each ADK model request inside a slice",
    )
    parser.add_argument(
        "--capture-root",
        default=str(REPO_ROOT / "eval_outputs" / "adk_single_request_captures"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "eval_outputs" / "adk_single_slice_12run.json"),
    )
    return parser.parse_args()


def _load_ordered_cases(gt_path: Path, *, limit: int, offset: int) -> list[tuple[str, float]]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    ordered = sorted(gt.items(), key=lambda kv: kv[1].get("slice_index", 0))
    if len(ordered) <= limit:
        selected = ordered[offset:offset + limit]
    else:
        pool = ordered[offset:]
        if len(pool) <= limit:
            selected = pool
        else:
            step = (len(pool) - 1) / max(limit - 1, 1)
            selected = [pool[round(i * step)] for i in range(limit)]
    return [(filename, float(entry["ap_mm"])) for filename, entry in selected]


async def _run_case(
    *,
    image_path: Path,
    gt_mm: float,
    atlas: str,
    model: str,
    media_resolution: str,
    multimodal_history: str,
    max_iterations: int,
    capture_dir: Path,
) -> dict:
    os.environ["LANGSLICE_ADK_CAPTURE_REQUESTS_DIR"] = str(capture_dir)
    with Image.open(image_path) as raw:
        image = raw.copy()
    started = time.time()
    result = await run_single_slice_session(
        image=image,
        atlas_name=atlas,
        plane="coronal",
        model=model,
        max_iterations=max_iterations,
        max_retries=2,
        media_resolution=media_resolution,
        apply_clahe=True,
        target_transport="auto",
        multimodal_history=multimodal_history,
    )
    elapsed = time.time() - started
    error = abs(result.position_mm - gt_mm)
    return {
        "filename": image_path.name,
        "ground_truth_mm": gt_mm,
        "estimated_mm": result.position_mm,
        "error_mm": error,
        "fallback": FALLBACK_PHRASE in result.reasoning,
        "elapsed_s": round(elapsed, 2),
        "reasoning": result.reasoning,
        "capture_dir": str(capture_dir),
    }


async def _main_async(args: argparse.Namespace) -> dict:
    image_dir = Path(args.images)
    gt_path = Path(args.ground_truth) if args.ground_truth else image_dir / "ground_truth.json"
    cases = _load_ordered_cases(gt_path, limit=args.limit, offset=args.offset)
    records: list[dict] = []

    capture_root = Path(args.capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)
    if args.model_call_sleep > 0:
        os.environ["LANGSLICE_ADK_MODEL_CALL_DELAY_S"] = str(args.model_call_sleep)

    for idx, (filename, gt_mm) in enumerate(cases, start=1):
        if idx > 1 and args.sleep > 0:
            time.sleep(args.sleep)
        capture_dir = capture_root / f"{idx:02d}_{Path(filename).stem}"
        print(f"[{idx}/{len(cases)}] {filename} GT={gt_mm:.3f}mm", file=sys.stderr)
        try:
            record = await _run_case(
                image_path=image_dir / filename,
                gt_mm=gt_mm,
                atlas=args.atlas,
                model=args.model,
                media_resolution=args.media_resolution,
                multimodal_history=args.multimodal_history,
                max_iterations=args.max_iterations,
                capture_dir=capture_dir,
            )
        except Exception as exc:
            record = {
                "filename": filename,
                "ground_truth_mm": gt_mm,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "capture_dir": str(capture_dir),
            }
        records.append(record)

    errors = [float(r["error_mm"]) for r in records if "error_mm" in r]
    summary = {
        "n_slices": len(records),
        "n_failures": sum(1 for r in records if r.get("status") == "error"),
        "n_fallbacks": sum(1 for r in records if r.get("fallback")),
        "mae_mm": sum(errors) / len(errors) if errors else None,
        "max_error_mm": max(errors) if errors else None,
    }
    return {
        "task": "adk_single_slice_transport_eval",
        "model": args.model,
        "atlas": args.atlas,
        "media_resolution": args.media_resolution,
        "thinking_level": "MEDIUM",
        "target_transport": "auto",
        "multimodal_history": args.multimodal_history,
        "tools": ["fetch_atlas", "submit_estimate"],
        "preprocessing": "runner normalize -> downscale -> CLAHE",
        "limit": args.limit,
        "slices": records,
        "summary": summary,
    }


def main() -> int:
    args = _parse_args()
    output = asyncio.run(_main_async(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved: {out_path}")
    return 0 if output["summary"]["n_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
