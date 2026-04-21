"""Phase 6 Task 6.2 smoke test: observe side_by_side-tool invocations in a group run.

Mirrors ``eval/smoke_group.py`` but drives the ADK group agent directly
(instead of through ``estimate_group``) so we can tap the event stream and
count function calls by tool name. The prod runner does not expose per-event
tool counts and the plan calls for a one-off observability check, so the
harness is duplicated here rather than adding a prod API just for this.

Gates:
- Hard: runs without exceptions, no fallback triggered.
- Soft: at least one ``side_by_side`` tool call observed during the run. If
  the soft gate fails the script returns exit 2 so the caller can decide
  whether to invest in prompt tweaks — we do not iterate on prompts here.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
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

from google.adk.apps.app import App  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from PIL import Image  # noqa: E402

from langslice import vlm_config  # noqa: E402
from langslice.atlas.core import (  # noqa: E402
    get_in_plane_long_edge,
    get_position_range_mm,
    load_atlas,
)
from langslice.harness.estimation.adk_plugins import (  # noqa: E402
    PersistentMultimodalToolResultsPlugin,
)
from langslice.harness.estimation.group import build_group_agent  # noqa: E402
from langslice.harness.estimation.runner import _encode_target_part  # noqa: E402
from langslice.harness.estimation.session import (  # noqa: E402
    build_initial_state,
    target_key,
)

TEST_IMAGES_DIR = Path(r"C:\LabSoftware\LangSlice\references\TestImages\M01")
GT_PATH = TEST_IMAGES_DIR / "ground_truth.json"
MODEL = "gemini-3-flash-preview"
ATLAS = "allen_mouse_25um"
APP_NAME = "langslice-smoke-sbs"
USER_ID = "smoke"

# Same mid-brain group as smoke_group.py.
SLICES = [
    "M01_002_005.tif",  # 5.620 mm  (slice_index 13)
    "M01_002_006.tif",  # 5.734 mm  (slice_index 14)
    "M01_002_007.tif",  # 5.931 mm  (slice_index 15)
    "M01_002_008.tif",  # 6.133 mm  (slice_index 16)
]


async def run_smoke() -> dict:
    gt = json.loads(GT_PATH.read_text())
    images = [Image.open(TEST_IMAGES_DIR / name) for name in SLICES]
    gts = [float(gt[name]["ap_mm"]) for name in SLICES]
    n_slices = len(images)
    gaps = [abs(gts[i + 1] - gts[i]) for i in range(n_slices - 1)]
    interval_mm = sum(gaps) / len(gaps)

    atlas = load_atlas(ATLAS)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane="coronal")
    atlas_long_edge = get_in_plane_long_edge(atlas, plane="coronal")
    thinking_cfg = vlm_config.build_thinking_config(MODEL, "MEDIUM")
    species = str(atlas.metadata.get("species", "mouse"))

    agent = build_group_agent(
        atlas_name=ATLAS,
        plane="coronal",
        species=species,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        n_slices=n_slices,
        interval_mm=interval_mm,
        thickness_um=50,
        model=MODEL,
        thinking_config=thinking_cfg,
    )

    app = App(
        name=APP_NAME,
        root_agent=agent,
        plugins=[PersistentMultimodalToolResultsPlugin()],
    )
    runner = InMemoryRunner(app=app)
    assert runner.artifact_service is not None
    assert runner.session_service is not None

    max_iterations = 25
    initial_state = build_initial_state(
        atlas_name=ATLAS,
        plane="coronal",
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        n_slices=n_slices,
        interval_mm=interval_mm,
        thickness_um=50,
        max_iterations=max_iterations,
    )

    # Encode target parts once (same order as run_group_session).
    target_parts = [
        _encode_target_part(img, atlas_long_edge, apply_clahe=True) for img in images
    ]

    session_id = "smoke_sbs_attempt"
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state=dict(initial_state),
    )

    # Seed each target image as 'target:N' so the side_by_side tool can
    # reference them.
    for i, part in enumerate(target_parts):
        await runner.artifact_service.save_artifact(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
            filename=target_key(i + 1),
            artifact=part,
        )

    parts: list[types.Part] = []
    for i, part in enumerate(target_parts):
        parts.append(
            types.Part.from_text(
                text=f"Slice {i + 1} (artifact: '{target_key(i + 1)}'):"
            )
        )
        parts.append(part)
    parts.append(types.Part.from_text(text="Determine the position of each slice."))
    new_message = types.Content(role="user", parts=parts)

    tool_counts: Counter[str] = Counter()
    tool_calls_log: list[dict] = []
    tool_call_count = 0
    capped = False

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=new_message,
    ):
        fcs = event.get_function_calls() or []
        for fc in fcs:
            tool_counts[fc.name] += 1
            tool_calls_log.append(
                {"name": fc.name, "args": dict(fc.args) if fc.args else {}}
            )
        tool_call_count += len(fcs)
        if tool_call_count > max_iterations:
            capped = True
            break

        current = await runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        if current is not None and current.state.get("result") is not None:
            break

    final = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    result = final.state.get("result") if final is not None else None

    return {
        "tool_counts": dict(tool_counts),
        "tool_calls_log": tool_calls_log,
        "result": result,
        "gts": gts,
        "capped": capped,
    }


def main() -> int:
    fallback_phrase = "Model did not submit"
    print(f"Group: {len(SLICES)} mid-brain slices from M01 wellplate 2", flush=True)
    print(f"Model: {MODEL}  Atlas: {ATLAS}", flush=True)
    print("Running ADK group agent (direct runner) and counting tool calls...",
          flush=True)

    t0 = time.time()
    try:
        outcome = asyncio.run(run_smoke())
    except Exception as exc:  # noqa: BLE001 — smoke test captures everything
        elapsed = time.time() - t0
        print(f"ERROR: {type(exc).__name__}: {exc} ({elapsed:.1f}s)")
        summary = {
            "task": "6.2 side_by_side smoke",
            "model": MODEL,
            "atlas": ATLAS,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "elapsed_s": elapsed,
            },
        }
        out_dir = REPO_ROOT / "eval_outputs"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "post_step6_sbs_M01.json").write_text(
            json.dumps(summary, indent=2)
        )
        return 1

    elapsed = time.time() - t0
    result = outcome["result"]
    tool_counts = outcome["tool_counts"]
    tool_calls_log = outcome["tool_calls_log"]
    gts = outcome["gts"]
    capped = outcome["capped"]

    n_errors = 0
    fallback = False
    max_err = float("nan")
    mae = float("nan")
    per_slice: list[dict] = []
    reasoning_str = ""

    if result is not None:
        predicted = [float(p) for p in result.get("positions_mm", [])]
        reasoning_str = str(result.get("reasoning", ""))
        fallback = fallback_phrase in reasoning_str
        if len(predicted) == len(SLICES):
            errors = [abs(p - g) for p, g in zip(predicted, gts, strict=True)]
            mae = sum(errors) / len(errors)
            max_err = max(errors)
            per_slice = [
                {
                    "name": SLICES[i],
                    "gt_mm": gts[i],
                    "predicted_mm": predicted[i],
                    "error_mm": errors[i],
                }
                for i in range(len(SLICES))
            ]
    else:
        n_errors = 1
        fallback = True

    sbs_calls = tool_counts.get("side_by_side", 0)
    gate_pass_hard = n_errors == 0 and not fallback and not capped
    gate_pass_soft = sbs_calls >= 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for rec in per_slice:
        print(
            f"  {rec['name']}: pred={rec['predicted_mm']:.3f} "
            f"gt={rec['gt_mm']:.3f} err={rec['error_mm']:.3f} mm"
        )
    print(f"\nTool counts:       {dict(tool_counts)}")
    print(f"side_by_side calls: {sbs_calls}")
    print(f"MAE:               {mae:.3f} mm" if mae == mae else "MAE:               n/a")
    print(
        f"Max error:         {max_err:.3f} mm"
        if max_err == max_err
        else "Max error:         n/a"
    )
    print(f"Fallback:          {fallback}")
    print(f"Capped:            {capped}")
    print(f"Wall time:         {elapsed:.1f} s")
    print(f"\nHard gate:         {'PASS' if gate_pass_hard else 'FAIL'}")
    print(f"Soft gate:         {'PASS' if gate_pass_soft else 'FAIL'}")

    summary = {
        "task": "6.2 side_by_side smoke",
        "model": MODEL,
        "atlas": ATLAS,
        "slices": per_slice,
        "summary": {
            "n_slices": len(SLICES),
            "fallback": fallback,
            "capped": capped,
            "mae_mm": mae,
            "max_error_mm": max_err,
            "wall_time_s": round(elapsed, 1),
            "tool_counts": dict(tool_counts),
            "side_by_side_invocations": sbs_calls,
            "gate_pass_hard": gate_pass_hard,
            "gate_pass_soft": gate_pass_soft,
        },
        "tool_calls_log": tool_calls_log,
        "reasoning": reasoning_str,
    }

    out_dir = REPO_ROOT / "eval_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "post_step6_sbs_M01.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")

    if not gate_pass_hard:
        return 1
    if not gate_pass_soft:
        print(
            "WARN: side_by_side invocations == 0; soft gate failed. "
            "Consider prompt tweaks."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
