"""Run the legacy /main Gemini SDK single-slice estimator on an M01 subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


FALLBACK_PHRASES = ("did not submit", "falling back", "fallback")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=r"C:\LabSoftware\LangSlice")
    parser.add_argument("--images", default=None)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--atlas", default="allen_mouse_25um")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--media-resolution", default="medium")
    parser.add_argument("--thinking-level", default="MEDIUM")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--apply-clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CLAHE before calling the legacy estimator.",
    )
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent
            / "eval_outputs"
            / "main_sdk_single_12run.json"
        ),
    )
    return parser.parse_args()


def _load_ordered_cases(
    gt_path: Path, *, limit: int, offset: int
) -> list[tuple[str, float]]:
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


def _summarize(records: list[dict]) -> dict[str, float | int | None]:
    errors = [float(r["error_mm"]) for r in records if "error_mm" in r]
    return {
        "n_slices": len(records),
        "n_failures": sum(1 for r in records if r.get("status") == "error"),
        "n_fallbacks": sum(1 for r in records if r.get("fallback")),
        "mae_mm": (sum(errors) / len(errors)) if errors else None,
        "max_error_mm": max(errors) if errors else None,
    }


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["summary"] = _summarize(payload["slices"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from dotenv import load_dotenv
    from PIL import Image

    import langslice.vlm_config as vlm_config
    from langslice.atlas.core import get_coronal_long_edge, load_atlas
    from langslice.estimation.google.ap_single_slice import estimate_position
    from langslice.image_prep import (
        adaptive_preprocess,
        normalize_image,
        prepare_image_for_vlm,
    )

    load_dotenv(repo_root / ".env")
    os.environ.pop("LANGSLICE_VLM_DEBUG_DIR", None)
    vlm_config.set_model_name(args.model)
    vlm_config.set_thinking_level(args.thinking_level)

    image_dir = (
        Path(args.images)
        if args.images
        else repo_root / "references" / "TestImages" / "M01"
    )
    gt_path = (
        Path(args.ground_truth)
        if args.ground_truth
        else image_dir / "ground_truth.json"
    )
    out_path = Path(args.out)
    records: list[dict] = []
    payload = {
        "task": "main_sdk_single_slice_eval",
        "repo_root": str(repo_root),
        "model": args.model,
        "atlas": args.atlas,
        "media_resolution": args.media_resolution,
        "thinking_level": args.thinking_level,
        "preprocessing": (
            "external normalize -> downscale -> CLAHE, then main SDK loop"
            if args.apply_clahe
            else "main normalize -> downscale"
        ),
        "target_transport": "Gemini File API",
        "atlas_transport": "individual inline images from fetch_atlas",
        "limit": args.limit,
        "slices": records,
        "summary": {},
    }
    _write_payload(out_path, payload)

    cases = _load_ordered_cases(gt_path, limit=args.limit, offset=args.offset)
    atlas = load_atlas(args.atlas)
    atlas_long_edge = get_coronal_long_edge(atlas)
    for idx, (filename, gt_mm) in enumerate(cases, start=1):
        image_path = image_dir / filename
        print(f"[{idx}/{len(cases)}] {filename} GT={gt_mm:.3f}mm", file=sys.stderr, flush=True)
        started = time.time()
        try:
            with Image.open(image_path) as raw:
                image = raw.copy()
            if args.apply_clahe:
                image = normalize_image(image)
                image = prepare_image_for_vlm(
                    image, max_long_edge=atlas_long_edge
                ).image
                image = adaptive_preprocess(image)
            result = estimate_position(
                image=image,
                atlas_name=args.atlas,
                max_iterations=args.max_iterations,
                media_resolution=args.media_resolution,
                model_name=args.model,
                send_individually=True,
                on_progress=lambda msg: print(
                    f"[{idx}/{len(cases)}] {msg}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
            reasoning = str(result.reasoning)
            record = {
                "filename": filename,
                "ground_truth_mm": gt_mm,
                "estimated_mm": float(result.position_mm),
                "error_mm": abs(float(result.position_mm) - gt_mm),
                "fallback": any(p in reasoning.lower() for p in FALLBACK_PHRASES),
                "elapsed_s": round(time.time() - started, 2),
                "reasoning": reasoning,
            }
            print(
                f"[{idx}/{len(cases)}] OK est={record['estimated_mm']:.3f} "
                f"err={record['error_mm']:.3f} elapsed={record['elapsed_s']}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            record = {
                "filename": filename,
                "ground_truth_mm": gt_mm,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.time() - started, 2),
                "traceback": traceback.format_exc(limit=8),
            }
            print(f"[{idx}/{len(cases)}] ERROR {record['error']}", file=sys.stderr, flush=True)
        records.append(record)
        _write_payload(out_path, payload)

    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved: {out_path}")
    return 0 if payload["summary"]["n_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
