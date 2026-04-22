"""Compare legacy /main and ADK single-slice Gemini behavior in one window.

This is a quota/debug canary, not a full accuracy benchmark. It alternates the
legacy google.genai loop and the ADK loop on the same slice so we can see
whether RESOURCE_EXHAUSTED errors are global or specific to the ADK request
path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

import langslice.vlm_config as vlm_config  # noqa: E402
from langslice.estimation.google.ap_single_slice import (  # noqa: E402
    estimate_position as estimate_position_main,
)
from langslice.harness.estimation.runner import run_single_slice_session  # noqa: E402
from langslice.image_prep import adaptive_preprocess  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        default=str(REPO_ROOT.parent.parent / "references" / "TestImages" / "M01"),
        help="Directory containing M01 slice images",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Ground truth JSON path; defaults to <images>/ground_truth.json",
    )
    parser.add_argument("--slice", default="M01_002_008.tif")
    parser.add_argument("--atlas", default="allen_mouse_25um")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--media-resolution", default="medium")
    parser.add_argument("--thinking-level", default="MEDIUM")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--adk-max-retries", type=int, default=2)
    parser.add_argument("--between-run-sleep", type=float, default=45.0)
    parser.add_argument(
        "--model-call-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep before each ADK model call.",
    )
    parser.add_argument(
        "--main-apply-clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preprocess the legacy /main input with CLAHE before calling it.",
    )
    parser.add_argument(
        "--adk-apply-clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CLAHE inside the ADK runner.",
    )
    parser.add_argument(
        "--capture-root",
        default=str(REPO_ROOT / "eval_outputs" / "main_vs_adk_canary_captures"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "eval_outputs" / "main_vs_adk_canary.json"),
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Print the legacy /main progress callback.",
    )
    return parser.parse_args()


def _load_ground_truth(gt_path: Path, filename: str) -> float | None:
    if not gt_path.exists():
        return None
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    entry = data.get(filename)
    if not isinstance(entry, dict):
        return None
    value = entry.get("ap_mm")
    return float(value) if value is not None else None


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as raw:
        return raw.copy()


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _result_record(
    *,
    impl: str,
    run_index: int,
    gt_mm: float | None,
    started_at: float,
    status: str,
    position_mm: float | None = None,
    reasoning: str | None = None,
    error: str | None = None,
    capture_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed_s = round(time.time() - started_at, 2)
    record: dict[str, Any] = {
        "impl": impl,
        "run_index": run_index,
        "status": status,
        "elapsed_s": elapsed_s,
    }
    if position_mm is not None:
        record["position_mm"] = position_mm
        if gt_mm is not None:
            record["ground_truth_mm"] = gt_mm
            record["error_mm"] = abs(position_mm - gt_mm)
    if reasoning is not None:
        record["reasoning"] = reasoning
    if error is not None:
        record["error"] = error
    if capture_dir is not None:
        record["capture_dir"] = str(capture_dir)
    if extra:
        record.update(extra)
    return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for impl in ("main", "adk"):
        impl_records = [r for r in records if r.get("impl") == impl]
        errors = [
            float(r["error_mm"])
            for r in impl_records
            if r.get("status") == "ok" and "error_mm" in r
        ]
        summary[impl] = {
            "n": len(impl_records),
            "ok": sum(1 for r in impl_records if r.get("status") == "ok"),
            "failures": sum(1 for r in impl_records if r.get("status") != "ok"),
            "mae_mm": (sum(errors) / len(errors)) if errors else None,
            "max_error_mm": max(errors) if errors else None,
            "resource_exhausted": sum(
                1
                for r in impl_records
                if "RESOURCE_EXHAUSTED" in str(r.get("error", ""))
                or "429" in str(r.get("error", ""))
            ),
        }
    return summary


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["summary"] = _summarize(payload["records"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_main(
    *,
    image_path: Path,
    gt_mm: float | None,
    args: argparse.Namespace,
    run_index: int,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    image = _load_image(image_path)
    if args.main_apply_clahe:
        image = adaptive_preprocess(image)

    started = time.time()
    try:
        result = estimate_position_main(
            image=image,
            atlas_name=args.atlas,
            on_progress=progress,
            max_iterations=args.max_iterations,
            media_resolution=args.media_resolution,
            model_name=args.model,
            send_individually=True,
        )
        return _result_record(
            impl="main",
            run_index=run_index,
            gt_mm=gt_mm,
            started_at=started,
            status="ok",
            position_mm=float(result.position_mm),
            reasoning=str(result.reasoning),
            extra={"main_apply_clahe": bool(args.main_apply_clahe)},
        )
    except Exception as exc:
        return _result_record(
            impl="main",
            run_index=run_index,
            gt_mm=gt_mm,
            started_at=started,
            status="error",
            error=_error_text(exc),
            extra={
                "main_apply_clahe": bool(args.main_apply_clahe),
                "traceback": traceback.format_exc(limit=8),
            },
        )


async def _run_adk(
    *,
    image_path: Path,
    gt_mm: float | None,
    args: argparse.Namespace,
    run_index: int,
    capture_dir: Path,
) -> dict[str, Any]:
    image = _load_image(image_path)
    os.environ["LANGSLICE_ADK_CAPTURE_REQUESTS_DIR"] = str(capture_dir)
    if args.model_call_sleep > 0:
        os.environ["LANGSLICE_ADK_MODEL_CALL_DELAY_S"] = str(args.model_call_sleep)
    else:
        os.environ.pop("LANGSLICE_ADK_MODEL_CALL_DELAY_S", None)

    started = time.time()
    try:
        result = await run_single_slice_session(
            image=image,
            atlas_name=args.atlas,
            plane="coronal",
            model=args.model,
            max_iterations=args.max_iterations,
            max_retries=args.adk_max_retries,
            media_resolution=args.media_resolution,
            thinking_level=args.thinking_level,
            apply_clahe=args.adk_apply_clahe,
            target_transport="auto",
            multimodal_history="persistent",
        )
        return _result_record(
            impl="adk",
            run_index=run_index,
            gt_mm=gt_mm,
            started_at=started,
            status="ok",
            position_mm=float(result.position_mm),
            reasoning=str(result.reasoning),
            capture_dir=capture_dir,
            extra={
                "adk_apply_clahe": bool(args.adk_apply_clahe),
                "adk_max_retries": int(args.adk_max_retries),
                "model_call_sleep_s": float(args.model_call_sleep),
            },
        )
    except Exception as exc:
        return _result_record(
            impl="adk",
            run_index=run_index,
            gt_mm=gt_mm,
            started_at=started,
            status="error",
            error=_error_text(exc),
            capture_dir=capture_dir,
            extra={
                "adk_apply_clahe": bool(args.adk_apply_clahe),
                "adk_max_retries": int(args.adk_max_retries),
                "model_call_sleep_s": float(args.model_call_sleep),
                "traceback": traceback.format_exc(limit=8),
            },
        )


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    vlm_config.set_model_name(args.model)
    vlm_config.set_thinking_level(args.thinking_level)

    image_dir = Path(args.images)
    image_path = image_dir / args.slice
    gt_path = Path(args.ground_truth) if args.ground_truth else image_dir / "ground_truth.json"
    gt_mm = _load_ground_truth(gt_path, args.slice)
    capture_root = Path(args.capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "task": "main_vs_adk_single_slice_quota_canary",
        "slice": args.slice,
        "ground_truth_mm": gt_mm,
        "atlas": args.atlas,
        "model": args.model,
        "media_resolution": args.media_resolution,
        "thinking_level": args.thinking_level,
        "pairs": args.pairs,
        "max_iterations": args.max_iterations,
        "order": "main_then_adk_repeated",
        "records": [],
        "summary": {},
    }
    out_path = Path(args.out)

    def progress(message: str) -> None:
        if args.verbose_progress:
            print(f"[main progress] {message}", file=sys.stderr, flush=True)

    for pair_idx in range(1, args.pairs + 1):
        print(f"[{pair_idx}/{args.pairs}] /main starting", file=sys.stderr, flush=True)
        main_record = _run_main(
            image_path=image_path,
            gt_mm=gt_mm,
            args=args,
            run_index=pair_idx,
            progress=progress,
        )
        payload["records"].append(main_record)
        _write_output(out_path, payload)
        print(
            f"[{pair_idx}/{args.pairs}] /main {main_record['status']} "
            f"elapsed={main_record['elapsed_s']}s",
            file=sys.stderr,
            flush=True,
        )

        if args.between_run_sleep > 0:
            time.sleep(args.between_run_sleep)

        capture_dir = capture_root / f"pair_{pair_idx:02d}_adk"
        print(f"[{pair_idx}/{args.pairs}] ADK starting", file=sys.stderr, flush=True)
        adk_record = await _run_adk(
            image_path=image_path,
            gt_mm=gt_mm,
            args=args,
            run_index=pair_idx,
            capture_dir=capture_dir,
        )
        payload["records"].append(adk_record)
        _write_output(out_path, payload)
        print(
            f"[{pair_idx}/{args.pairs}] ADK {adk_record['status']} "
            f"elapsed={adk_record['elapsed_s']}s",
            file=sys.stderr,
            flush=True,
        )

        if pair_idx < args.pairs and args.between_run_sleep > 0:
            time.sleep(args.between_run_sleep)

    return payload


def main() -> int:
    args = _parse_args()
    payload = asyncio.run(_main_async(args))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved: {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
