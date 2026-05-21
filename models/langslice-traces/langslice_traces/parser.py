"""Parse SFT-corpus JSONL rows into CanonicalTrace instances."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import CanonicalTrace, FinalAnswer, Plane, ToolStep

_VALID_PLANES: tuple[str, ...] = ("coronal", "sagittal", "horizontal")


def parse_canonical_trace(
    row: dict[str, Any],
    *,
    dataset_root: Path | None = None,
) -> CanonicalTrace:
    """Convert one SFT JSONL row dict into a CanonicalTrace.

    ``dataset_root`` is recorded on the returned trace and used by renderers
    to resolve relative ``query_image_paths`` and tool-result image paths.

    Raises
    ------
    ValueError
        If ``row["plane"]`` is not one of "coronal", "sagittal", "horizontal".
    KeyError
        If a required top-level field is missing.
    """
    plane_raw = row["plane"]
    if plane_raw not in _VALID_PLANES:
        raise ValueError(
            f"plane must be one of {_VALID_PLANES!r} (got {plane_raw!r})"
        )
    plane: Plane = plane_raw  # type: ignore[assignment]

    trace_steps = row["trace"]
    if not isinstance(trace_steps, list) or not trace_steps:
        raise ValueError("trace must be a non-empty list")

    # Walk every step except the last; each must be a tool_call/tool_result pair.
    tool_steps: list[ToolStep] = []
    for i, step in enumerate(trace_steps[:-1]):
        if "tool_call" not in step or "tool_result" not in step:
            raise ValueError(
                f"trace[{i}] must have both 'tool_call' and 'tool_result'"
            )
        tool_call = step["tool_call"]
        tool_result = step["tool_result"]
        tool_steps.append(
            ToolStep(
                call_name=tool_call["name"],
                call_args=dict(tool_call.get("args", {})),
                result_image_paths=list(tool_result.get("image_paths", [])),
                result_text=tool_result.get("text", ""),
            )
        )

    # Terminal step must be the submit.
    last = trace_steps[-1]
    if "submit" not in last:
        raise ValueError("trace must end with a 'submit' step")
    submit = last["submit"]
    submit_args = submit.get("args", {})
    final_answer = FinalAnswer(
        name=submit["name"],
        position_mm=float(submit_args["position_mm"]),
        reasoning=submit_args["reasoning"],
    )

    return CanonicalTrace(
        atlas_name=row["atlas_name"],
        atlas_version=row["atlas_version"],
        plane=plane,
        subject_id=row["subject_id"],
        system_prompt_kind=row["system_prompt_kind"],
        bucket=row["bucket"],
        query_image_paths=list(row["query_image_paths"]),
        user_prompt_text=row["user_prompt_text"],
        tool_steps=tool_steps,
        final_answer=final_answer,
        quality=dict(row.get("quality", {})),
        gemini_reasoning=row.get("gemini_reasoning"),
        dataset_root=dataset_root,
    )


def iter_canonical_traces(jsonl_path: Path) -> Iterator[CanonicalTrace]:
    """Yield CanonicalTrace per line in an SFT-corpus JSONL file.

    Sets ``dataset_root`` on each trace to ``jsonl_path.parent`` so renderers
    can hydrate images via relative paths.
    """
    path = Path(jsonl_path)
    root = path.parent
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            yield parse_canonical_trace(row, dataset_root=root)
