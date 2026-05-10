"""Run one model through one SliceBench bench, write predictions.jsonl.

Calls the production agent loop (`run_single_slice_session`) directly with an
`AgentTraceRecorder` plugin so token-usage and per-call cost are captured.
Resumable: if ``predictions.jsonl`` exists in the output dir, sections already
completed are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# Ensure `slicebench` is importable when invoked as `python slicebench/run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _route_api_key(env_var_name: str | None) -> None:
    """Override GEMINI_API_KEY with a sibling var (e.g. GEMINI_API_KEY_A) BEFORE
    the harness loads dotenv. Lets us route runs through specific AI Studio
    projects without editing .env."""
    if not env_var_name:
        return
    # Load .env first so GEMINI_API_KEY_A/B are available in the process env.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    value = os.environ.get(env_var_name)
    if not value:
        raise SystemExit(
            f"--api-key-var {env_var_name!r} requested but that env var is not set. "
            f"Add it to .env or export it before launching."
        )
    os.environ["GEMINI_API_KEY"] = value
    print(f"  api key: routed via ${env_var_name}")


# Parse --api-key-var early so the override happens before importing the harness
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--api-key-var", default=None)
_pre_args, _ = _pre_parser.parse_known_args()
_route_api_key(_pre_args.api_key_var)


from PIL import Image

from langslice_harness.harness.estimation.runner import run_single_slice_session
from langslice_harness.harness.estimation.trace_collection import (
    AgentTraceRecorder,
    estimate_cost_usd,
)

from slicebench.loader import EvalRow, REPO_ROOT, load_bench
from slicebench.pricing import pricing_for


# ──────────────────────────────────────────────────────────────────────────
# Resumability
# ──────────────────────────────────────────────────────────────────────────


def _load_completed_section_ids(path: Path) -> set[tuple[str, int]]:
    """Return the set of (section_id, generation_idx) pairs already in predictions.

    For pre-num_generations runs the field is absent; we treat those rows as
    generation 0 so the resumability semantics stay sane.
    """
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            sid = row.get("section_id")
            if sid:
                gen = int(row.get("generation_idx", 0))
                done.add((sid, gen))
    return done


# ──────────────────────────────────────────────────────────────────────────
# Single-row execution
# ──────────────────────────────────────────────────────────────────────────


def _run_one(
    row: EvalRow,
    model: str,
    run_id: str,
    *,
    generation_idx: int = 0,
    temperature: float | None = None,
) -> dict:
    """Run estimate_position once. Returns a result dict for predictions.jsonl."""
    image_abs = (REPO_ROOT / row.image_path).resolve()
    pricing = pricing_for(model)
    recorder = AgentTraceRecorder()
    started = time.time()
    try:
        with Image.open(image_abs) as raw:
            image = raw.copy()
        kwargs: dict[str, object] = {
            "image": image,
            "atlas_name": row.atlas,
            "plane": row.plane,
            "model": model,
            "trace_recorder": recorder,
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        result = asyncio.run(run_single_slice_session(**kwargs))  # type: ignore[arg-type]
        predicted_mm = float(result.position_mm)
        reasoning = (result.reasoning or "")[:2000]
        error_msg = None
    except Exception as exc:  # noqa: BLE001
        predicted_mm = None
        reasoning = ""
        error_msg = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    usage = {k: v for k, v in recorder.usage_totals.items() if isinstance(v, (int, float))}
    cost_usd = estimate_cost_usd(usage, pricing) if (pricing is not None and usage) else 0.0

    return {
        "section_id": row.section_id,
        "generation_idx": generation_idx,
        "dataset": row.dataset,
        "subject_id": row.subject_id,
        "plane": row.plane,
        "atlas": row.atlas,
        "species": row.species,
        "staining": row.staining,
        "imaging": row.imaging,
        "image_path": row.image_path,
        "plane_extent_mm": row.plane_extent_mm,
        "truth_mm": row.truth_mm,
        "predicted_mm": predicted_mm,
        "reasoning": reasoning,
        "error": error_msg,
        "elapsed_s": round(elapsed, 3),
        "usage": usage,
        "cost_usd": round(float(cost_usd), 6),
        "temperature": temperature,
        "model": model,
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def run(
    model: str,
    bench: str,
    out_dir: Path,
    *,
    concurrency: int = 1,
    limit: int | None = None,
    num_generations: int = 1,
    temperature: float | None = None,
) -> dict:
    if num_generations < 1:
        raise ValueError(f"num_generations must be >= 1, got {num_generations}")

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"

    rows = load_bench(bench)
    if limit is not None:
        rows = rows[:limit]

    completed = _load_completed_section_ids(predictions_path)
    # Expand each row × num_generations into a (row, gen_idx) work unit, then
    # filter out the (section_id, gen_idx) pairs already on disk for resumes.
    work: list[tuple[EvalRow, int]] = [
        (r, g) for r in rows for g in range(num_generations)
        if (r.section_id, g) not in completed
    ]
    n_total_units = len(rows) * num_generations
    pricing = pricing_for(model)
    pricing_note = (
        f"pricing=${pricing.input_per_million}/${pricing.output_per_million} per M tokens"
        if pricing else "pricing=free/local"
    )
    print(
        f"bench={bench} model={model} total={len(rows)} num_gens={num_generations} "
        f"units_total={n_total_units} units_done={len(completed)} units_todo={len(work)} "
        f"concurrency={concurrency}"
        + (f" temperature={temperature}" if temperature is not None else "")
    )
    print(f"  {pricing_note}")
    if not work:
        print("nothing to do; exiting")
        return {"n_total": len(rows), "n_done": len(completed), "n_added": 0}

    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc).isoformat()
    started_t = time.time()

    write_lock = Lock()
    running_cost = 0.0
    n_added = 0
    n_failed = 0

    def _append(result: dict) -> None:
        nonlocal running_cost
        with write_lock:
            with predictions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")
            running_cost += float(result.get("cost_usd") or 0.0)

    def _log(idx: int, total: int, row: EvalRow, gen: int, result: dict) -> None:
        gen_tag = f" gen={gen}" if num_generations > 1 else ""
        if result.get("error"):
            print(
                f"  [{idx}/{total}] FAIL {row.section_id}{gen_tag}: {result['error']}",
                file=sys.stderr,
            )
        else:
            cost_str = f"${result.get('cost_usd', 0):.4f}" if pricing else "free"
            print(
                f"  [{idx}/{total}] {row.section_id}{gen_tag} "
                f"pred={result['predicted_mm']:.3f} truth={row.truth_mm:.3f} "
                f"({result['elapsed_s']:.1f}s) {cost_str}  running=${running_cost:.4f}"
            )

    if concurrency <= 1:
        for i, (row, gen_idx) in enumerate(work, 1):
            result = _run_one(
                row, model, run_id,
                generation_idx=gen_idx, temperature=temperature,
            )
            _append(result)
            n_added += 1
            if result.get("error"):
                n_failed += 1
            _log(i, len(work), row, gen_idx, result)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _run_one, row, model, run_id,
                    generation_idx=gen_idx, temperature=temperature,
                ): (row, gen_idx)
                for row, gen_idx in work
            }
            done = 0
            for fut in as_completed(futures):
                row, gen_idx = futures[fut]
                done += 1
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    result = {
                        "section_id": row.section_id,
                        "generation_idx": gen_idx,
                        "error": f"{type(exc).__name__}: {exc}",
                        "predicted_mm": None,
                        "cost_usd": 0.0,
                        "elapsed_s": 0.0,
                    }
                _append(result)
                n_added += 1
                if result.get("error"):
                    n_failed += 1
                _log(done, len(work), row, gen_idx, result)

    elapsed_total = time.time() - started_t
    meta = {
        "run_id": run_id,
        "model": model,
        "bench": bench,
        "concurrency": concurrency,
        "num_generations": num_generations,
        "temperature": temperature,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_total, 1),
        "n_total": len(rows),
        "n_done_before": len(completed),
        "n_added": n_added,
        "n_failed": n_failed,
        "pricing_per_million": (
            {
                "input": pricing.input_per_million,
                "output": pricing.output_per_million,
                "cached_input": pricing.cached_input_per_million,
            }
            if pricing
            else None
        ),
        "estimated_cost_usd_this_run": round(running_cost, 6),
    }
    if meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["started_at"] = prior.get("started_at", meta["started_at"])
        meta["estimated_cost_usd_total"] = round(
            float(prior.get("estimated_cost_usd_total") or 0.0) + running_cost, 6
        )
        meta["n_added_total"] = int(prior.get("n_added_total") or 0) + n_added
        meta["n_failed_total"] = int(prior.get("n_failed_total") or 0) + n_failed
    else:
        meta["estimated_cost_usd_total"] = round(running_cost, 6)
        meta["n_added_total"] = n_added
        meta["n_failed_total"] = n_failed
    # Field used by plot.py for cost-vs-accuracy scatter
    meta["estimated_cost_usd"] = meta["estimated_cost_usd_total"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    avg_per_section = running_cost / max(1, n_added) if running_cost else 0.0
    print(
        f"\nwrote {n_added} predictions ({n_failed} failed) in {elapsed_total:.0f}s -> {predictions_path}"
    )
    print(
        f"cost this run: ${running_cost:.4f}  "
        f"(avg per section: ${avg_per_section:.4f})  "
        f"total across resumes: ${meta['estimated_cost_usd_total']:.4f}"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one model through SliceBench.")
    parser.add_argument(
        "--model", required=True,
        help=(
            "Model identifier; resolves through langslice_harness.model_resolver. "
            "Examples: gemini-3-flash-preview, gemma-4-31b-it, gemma-4-26b-a4b-it, "
            "litellm-proxy:<alias> (for llama.cpp)."
        ),
    )
    parser.add_argument("--bench", choices=["tiny", "small", "large"], required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Parallel workers. Default 1. Pro must stay 1; Flash tolerates 6-8.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap to first N rows (dev/smoke).")
    parser.add_argument(
        "--num-generations", type=int, default=1,
        help=(
            "Sample N completions per slice (default 1). With N>1 each section "
            "gets N rows in predictions.jsonl, distinguished by generation_idx."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help=(
            "Override sampling temperature (passed to run_single_slice_session). "
            "Useful with --num-generations to spread the rollouts; common values "
            "are 0.9 (RLVR-style) or 1.0 (harness default)."
        ),
    )
    parser.add_argument(
        "--api-key-var", default=None,
        help=(
            "Name of an env var to copy into GEMINI_API_KEY before the harness "
            "loads (e.g. GEMINI_API_KEY_A or GEMINI_API_KEY_B). Lets you route "
            "runs through specific AI Studio projects without editing .env."
        ),
    )
    args = parser.parse_args(argv)

    run(
        args.model, args.bench, args.out,
        concurrency=args.concurrency, limit=args.limit,
        num_generations=args.num_generations, temperature=args.temperature,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
