"""Convert ADK rollout output into a langslice-native SFT JSONL row.

This mirrors the distilled-corpus trace helpers, but keeps the implementation
inside the package tree so tests and installed training code do not depend on
gitignored ``_local`` scripts.

Two extra audit fields land in ``quality``:

    quality.source           = "expert_iteration"
    quality.iteration_round  = <round index, 0 for the MVP>

These are additive and ignored by the SFT validator; a downstream curator
can stratify or drop iterative rows by inspecting them.
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRACE_PACKAGE_ROOT = _REPO_ROOT / "models" / "langslice-traces"
if _TRACE_PACKAGE_ROOT.is_dir():
    trace_root = str(_TRACE_PACKAGE_ROOT)
    if trace_root not in sys.path:
        sys.path.insert(0, trace_root)
    existing = sys.modules.get("langslice_traces")
    existing_file = Path(getattr(existing, "__file__", "")).as_posix() if existing else ""
    if existing_file and trace_root.replace("\\", "/") not in existing_file:
        for module_name in [*sys.modules]:
            if module_name == "langslice_traces" or module_name.startswith("langslice_traces."):
                del sys.modules[module_name]

from langslice_traces.generator import _user_prompt
from langslice_traces.quality import classify_quality as _classify_quality
from langslice_traces.schema import Plane
from langslice_traces.trace_ops import trim_trace as _trim_trace

from langslice_harness.atlas.core import get_in_plane_long_edge, load_atlas
from langslice_harness.harness.estimation.trace_collection import (
    plane_rescue_threshold_mm,
    plane_tolerance_mm,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers re-exported for unit-test convenience
# ──────────────────────────────────────────────────────────────────────────


_VALID_PLANES: set[str] = {"coronal", "sagittal", "horizontal"}


def _plane(plane: str) -> Plane:
    if plane not in _VALID_PLANES:
        raise ValueError(f"unknown plane {plane!r}")
    return plane  # type: ignore[return-value]


def _tool_result_for(
    events: list[dict[str, Any]],
    start_idx: int,
    name: str,
) -> dict[str, Any] | None:
    for event in events[start_idx + 1:]:
        if event.get("role") == "assistant":
            return None
        if event.get("role") == "tool" and event.get("name") == name:
            return event
    return None


def _stage_atlas_artifact(
    artifact_path: str,
    *,
    run_dir: Path,
    out_dir: Path,
    atlas_name: str,
    plane: str,
) -> str:
    src = Path(artifact_path)
    if not src.is_absolute():
        src = run_dir / src
    src = src.resolve()

    dest_dir = out_dir / "atlas" / atlas_name / plane
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src != dest.resolve():
        shutil.copy2(src, dest)
    return dest.relative_to(out_dir).as_posix()


def _response_text(event: dict[str, Any] | None, positions: list[float]) -> str:
    response = (event or {}).get("response") or {}
    text = response.get("description") or response.get("text")
    if isinstance(text, str) and text.strip():
        return text
    joined = ", ".join(f"{p:.2f}" for p in positions)
    return f"Fetched {len(positions)} atlas sections at: {joined} mm"


def convert_trace(
    events: list[dict[str, Any]],
    *,
    run_dir: Path,
    atlas_name: str,
    plane: str,
    out_dir: Path,
) -> list[dict[str, Any]] | None:
    """Convert ADK recorder events to the compact SFT trace format."""
    trace: list[dict[str, Any]] = []

    for idx, event in enumerate(events):
        if event.get("role") != "assistant":
            continue
        for tool_call in event.get("tool_calls") or []:
            name = tool_call.get("name")
            args = dict(tool_call.get("args") or {})
            if name == "fetch_atlas":
                positions = [float(p) for p in args.get("positions_mm") or []]
                tool_event = _tool_result_for(events, idx, "fetch_atlas")
                image_paths = [
                    _stage_atlas_artifact(
                        str(artifact["path"]),
                        run_dir=run_dir,
                        out_dir=out_dir,
                        atlas_name=atlas_name,
                        plane=plane,
                    )
                    for artifact in (tool_event or {}).get("artifacts") or []
                    if artifact.get("type") == "image_path" and artifact.get("path")
                ]
                trace.append({
                    "tool_call": {
                        "name": "fetch_atlas",
                        "args": {"positions_mm": positions},
                    },
                    "tool_result": {
                        "image_paths": image_paths,
                        "text": _response_text(tool_event, positions),
                    },
                })
            elif name == "submit_estimate":
                trace.append({
                    "submit": {
                        "name": "submit_estimate",
                        "args": {
                            "position_mm": float(args["position_mm"]),
                            "reasoning": str(args.get("reasoning") or ""),
                        },
                    },
                })
                return trace

    return None


def classify_quality(
    *,
    model: str | None,
    err: float,
    tol: float,
    rescue: float | None,
    sub_canonical_mm: float,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return _classify_quality(
        model=model,
        err=err,
        tol=tol,
        rescue=rescue,
        sub_canonical_mm=sub_canonical_mm,
        events=events,
    )


def trim_trace(
    trace: list[dict[str, Any]],
    *,
    max_fetch_calls: int = 2,
    max_total_images: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _trim_trace(
        trace,
        max_fetch_calls=max_fetch_calls,
        max_total_images=max_total_images,
    )


def stage_query_image(
    src_rel: str,
    out_dir: Path,
    sid: str,
    *,
    atlas_name: str,
    plane: str,
    apply_clahe: bool,
    run_dir: Path | None = None,
) -> str:
    src_path = Path(src_rel)
    candidates: list[Path] = []
    if src_path.is_absolute():
        candidates.append(src_path)
    else:
        if run_dir is not None:
            candidates.append(run_dir / src_path)
        candidates.append(_REPO_ROOT / src_path)
        candidates.append(Path.cwd() / src_path)

    src = next((candidate for candidate in candidates if candidate.is_file()), None)
    if src is None:
        raise FileNotFoundError(f"could not stage missing query image {src_rel!r}")

    suffix = src.suffix or ".jpg"
    clahe = "_clahe" if apply_clahe else ""
    dest = out_dir / "queries" / f"{sid}{clahe}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest.relative_to(out_dir).as_posix()


def atlas_version(atlas_name: str) -> str:
    if atlas_name == "allen_mouse_25um":
        return "CCFv3"
    return ""


def user_prompt_text(plane: str, atlas_name: str) -> str:
    return _user_prompt(_plane(plane), atlas_name)


def estimate_tokens(
    *,
    n_query_images: int,
    n_atlas_images: int,
    n_fetch_calls: int,
    reasoning: str,
) -> int:
    return int(
        220
        + n_query_images * 800
        + n_atlas_images * 360
        + n_fetch_calls * 80
        + len(reasoning) / 4
    )


# ──────────────────────────────────────────────────────────────────────────
# Driver-facing entrypoint
# ──────────────────────────────────────────────────────────────────────────


def build_row(
    *,
    section_id: str,
    subject_id: str,
    atlas_name: str,
    plane: str,
    query_image_rel: str,
    events: list[dict[str, Any]],
    submitted_position_mm: float,
    sub_canonical_mm: float,
    abs_err_mm: float,
    model: str,
    iteration_round: int,
    out_dir: Path,
    run_dir: Path,
    apply_clahe: bool,
    max_fetch_calls: int = 2,
    max_total_images: int = 12,
) -> dict[str, Any] | None:
    """Construct one SFT-validatable row from a rollout's recorder state.

    Returns ``None`` when the trace lacks a terminal ``submit_estimate`` or
    when the trace cannot be reconstructed (caller should drop the rollout).

    Caller is responsible for staging the query image (so the ``out_dir``
    layout matches the distilled corpus). ``query_image_rel`` is the
    out_dir-relative path returned by ``stage_query_image``.
    """
    trace = convert_trace(
        events,
        run_dir=run_dir,
        atlas_name=atlas_name,
        plane=plane,
        out_dir=out_dir,
    )
    if not trace or "submit" not in trace[-1]:
        return None

    tol = plane_tolerance_mm(atlas_name, plane)
    rescue = plane_rescue_threshold_mm(atlas_name, plane)

    quality = classify_quality(
        model=model,
        err=abs_err_mm,
        tol=tol,
        rescue=rescue,
        sub_canonical_mm=sub_canonical_mm,
        events=events,
    )
    quality["preprocessing"] = "clahe" if apply_clahe else "raw"
    long_edge = get_in_plane_long_edge(load_atlas(atlas_name), plane=plane)
    quality["query_long_edge_px"] = long_edge

    if quality["accuracy"] in {"tight", "in_tolerance"}:
        quality["acceptance_tier"] = "strict"
    elif quality["accuracy"] == "rescued":
        quality["acceptance_tier"] = "rescued"
    else:
        quality["acceptance_tier"] = "out_of_tolerance"
    quality["rescue_threshold_mm"] = round(rescue, 4)

    # Audit-trail fields the spec calls out for iterative rows.
    quality["source"] = "expert_iteration"
    quality["iteration_round"] = iteration_round
    quality["submitted_position_mm"] = round(float(submitted_position_mm), 4)

    trace, trim_meta = trim_trace(
        trace,
        max_fetch_calls=max_fetch_calls,
        max_total_images=max_total_images,
    )
    quality["trim"] = trim_meta

    reasoning = ""
    if trace and "submit" in trace[-1]:
        reasoning = trace[-1]["submit"].get("args", {}).get("reasoning", "") or ""
    n_atlas = sum(
        len(s.get("tool_result", {}).get("image_paths") or []) for s in trace if "tool_call" in s
    )
    n_fetch = sum(1 for s in trace if "tool_call" in s)
    quality["estimated_tokens"] = estimate_tokens(
        n_query_images=1,
        n_atlas_images=n_atlas,
        n_fetch_calls=n_fetch,
        reasoning=reasoning,
    )

    return {
        "bucket": 1,
        "atlas_name": atlas_name,
        "atlas_version": atlas_version(atlas_name),
        "plane": plane,
        "subject_id": subject_id,
        # Phase 6: propagate section_id so train_sft.py's WeightedRandomSampler
        # can key curriculum weights by section_id rather than subject_id.
        # The iSFT real-rollout path provides section_id directly from the
        # allocation; real rollouts should always pass a non-empty value here.
        "section_id": section_id,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [query_image_rel],
        "user_prompt_text": user_prompt_text(plane, atlas_name),
        "trace": trace,
        "quality": quality,
    }
