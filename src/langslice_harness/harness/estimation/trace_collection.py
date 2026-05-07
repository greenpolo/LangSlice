"""Trace-collection helpers for SFT data generation."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from google.adk.plugins.base_plugin import BasePlugin
from PIL import Image

from langslice_harness.atlas.core import get_position_range_mm, load_atlas

TraceKind = Literal["single", "group"]


# Position-estimation tolerance is plane-relative (a fraction of the atlas
# extent on the slice-normal axis). Without this, a 0.9mm rescue threshold is
# 7% of coronal AP (13.2mm Allen) but 16% of canonical sagittal hemisphere
# (5.7mm) — too loose for the shorter planes. These percentages give roughly
# coronal-equivalent strictness across all three planes.
_TOLERANCE_PCT = 0.015  # accept threshold ≈ 0.20mm coronal Allen
_RESCUE_PCT = 0.07      # rescue threshold ≈ 0.92mm coronal Allen
_TOLERANCE_FLOOR_MM = 0.05


def plane_tolerance_mm(atlas_name: str, plane: str) -> float:
    """Atlas-aware accept tolerance for `categorize_trace`.

    For sagittal we use canonical-hemisphere extent (full ML extent / 2),
    matching the production constraint that the agent estimates within one
    hemisphere only.
    """
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    extent = hi
    if plane == "sagittal":
        extent = extent / 2.0
    return max(extent * _TOLERANCE_PCT, _TOLERANCE_FLOOR_MM)


def plane_rescue_threshold_mm(atlas_name: str, plane: str) -> float:
    """Atlas-aware rescue (near-miss) threshold."""
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    extent = hi
    if plane == "sagittal":
        extent = extent / 2.0
    return max(extent * _RESCUE_PCT, _TOLERANCE_FLOOR_MM * 4)


def canonicalize_positions(
    positions: list[float] | None, atlas_name: str, plane: str
) -> list[float] | None:
    """Map sagittal positions to canonical (single-hemisphere) form.

    The two hemispheres are mirror images so any GT (or model submission) in
    the upper-half ML range is equivalent to its mirror in the lower-half.
    Comparing GT and submission in canonical space removes the mirror flip
    that otherwise drives ~50% of sagittal "errors". No-op for coronal/horizontal.
    """
    if positions is None or plane != "sagittal" or not positions:
        return positions
    atlas = load_atlas(atlas_name)
    _, hi = get_position_range_mm(atlas, plane=plane)
    return [min(float(p), hi - float(p)) for p in positions]
SftExportMode = Literal["deployment", "rationale", "both"]
SingleRunner = Callable[..., Awaitable[Any]]
GroupRunner = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class TracePricing:
    """Per-million token pricing used for rough collection cost estimates."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0


@dataclass(frozen=True)
class TraceManifestRow:
    """One single-slice or grouped trace collection job."""

    id: str
    kind: TraceKind
    images: list[str]
    atlas: str
    plane: str
    truth_positions_mm: list[float]
    interval_um: int | None = None
    thickness_um: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


GEMINI_31_PRO_PREVIEW_PRICING = TracePricing(
    input_per_million=2.0,
    output_per_million=12.0,
    cached_input_per_million=0.20,
)

_PRIVATE_RESULT_KEYS = {"_langslice_multimodal_parts"}
_USAGE_FIELDS = (
    "prompt_token_count",
    "candidates_token_count",
    "total_token_count",
    "thoughts_token_count",
    "cached_content_token_count",
    "tool_use_prompt_token_count",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return _jsonable(dumped)
    return str(value)


def _usage_metadata_dict(response: Any) -> dict[str, int | float | str | bool]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    out: dict[str, int | float | str | bool] = {}
    for field_name in _USAGE_FIELDS:
        value = getattr(usage, field_name, None)
        if isinstance(value, (int, float, str, bool)):
            out[field_name] = value
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            for field_name in _USAGE_FIELDS:
                value = dumped.get(field_name)
                if isinstance(value, (int, float, str, bool)):
                    out[field_name] = value
    return out


class AgentTraceRecorder(BasePlugin):
    """ADK plugin that records deployment-shaped trace events and usage."""

    def __init__(
        self,
        *,
        name: str = "langslice_trace_recorder",
        tool_artifact_dir: str | Path | None = None,
    ) -> None:
        super().__init__(name)
        self.raw_events: list[dict[str, Any]] = []
        self.usage_by_call: list[dict[str, int | float | str | bool]] = []
        self.usage_totals: dict[str, int | float] = {}
        self.tool_call_count = 0
        self.tool_result_count = 0
        self.submit_rejections = 0
        self.tool_artifact_dir = Path(tool_artifact_dir) if tool_artifact_dir else None

    async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
        del callback_context
        self.record_model_response(llm_response)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> None:
        del tool_context
        self.record_tool_result(
            tool_name=str(getattr(tool, "name", None) or getattr(tool, "__name__", "tool")),
            tool_args=tool_args,
            result=result,
        )
        return None

    def record_model_response(self, response: Any) -> None:
        usage = _usage_metadata_dict(response)
        if usage:
            self.usage_by_call.append(usage)
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    self.usage_totals[key] = self.usage_totals.get(key, 0) + value

        content = getattr(response, "content", None)
        parts = getattr(content, "parts", None) or []
        visible_text: list[str] = []
        thought_summaries: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                if bool(getattr(part, "thought", False)):
                    thought_summaries.append(text)
                else:
                    visible_text.append(text)

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                args = getattr(function_call, "args", None) or {}
                tool_calls.append({
                    "name": str(getattr(function_call, "name", "")),
                    "args": _jsonable(dict(args)),
                })
                self.tool_call_count += 1

        if visible_text or thought_summaries or tool_calls or usage:
            self.raw_events.append({
                "role": "assistant",
                "visible_text": "\n".join(visible_text),
                "teacher_thought_summaries": thought_summaries,
                "tool_calls": tool_calls,
                "usage_metadata": usage,
            })

    def record_tool_result(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.tool_result_count += 1
        artifacts = self._persist_multimodal_tool_artifacts(tool_name, result)
        public_result = {
            key: value for key, value in result.items() if key not in _PRIVATE_RESULT_KEYS
        }
        if public_result.get("status") == "error" and tool_name.startswith("submit"):
            self.submit_rejections += 1
        event = {
            "role": "tool",
            "name": tool_name,
            "args": _jsonable(tool_args),
            "response": _jsonable(public_result),
        }
        if artifacts:
            event["artifacts"] = artifacts
        self.raw_events.append(event)

    def _persist_multimodal_tool_artifacts(
        self, tool_name: str, result: dict[str, Any]
    ) -> list[dict[str, str]]:
        if self.tool_artifact_dir is None:
            return []
        parts = result.get("_langslice_multimodal_parts")
        if not isinstance(parts, list):
            return []

        self.tool_artifact_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, str]] = []
        image_index = 0
        tool_slug = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tool_name
        ).strip("_") or "tool"
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            mime_type = getattr(inline_data, "mime_type", None)
            if not isinstance(data, bytes) or not isinstance(mime_type, str):
                continue
            if not mime_type.startswith("image/"):
                continue
            image_index += 1
            ext = ".png" if mime_type == "image/png" else ".jpg"
            filename = f"{tool_slug}_{self.tool_result_count:03d}_img{image_index:02d}{ext}"
            path = self.tool_artifact_dir / filename
            path.write_bytes(data)
            try:
                stored_path = path.relative_to(self.tool_artifact_dir.parent)
            except ValueError:
                stored_path = path
            artifacts.append({
                "type": "image_path",
                "path": str(stored_path).replace("\\", "/"),
                "mime_type": mime_type,
            })
        return artifacts


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest record requires non-empty string field {key!r}")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Manifest record requires numeric field {key!r}")
    return float(value)


def _require_float_list(data: dict[str, Any], key: str) -> list[float]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest record requires non-empty list field {key!r}")
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Manifest field {key!r} must contain only numbers")
        out.append(float(item))
    return out


def _optional_int(data: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Manifest field {key!r} must be an integer")
    return value


def _metadata(data: dict[str, Any], known_keys: set[str]) -> dict[str, Any]:
    explicit = data.get("metadata")
    out = dict(explicit) if isinstance(explicit, dict) else {}
    for key, value in data.items():
        if key not in known_keys:
            out[key] = value
    return out


def parse_manifest_record(data: dict[str, Any]) -> TraceManifestRow:
    """Parse and validate one JSON manifest record."""

    record_id = _require_str(data, "id")
    kind = _require_str(data, "kind")
    if kind not in {"single", "group"}:
        raise ValueError(f"Manifest record {record_id!r} has unsupported kind {kind!r}")

    atlas = _require_str(data, "atlas")
    plane = _require_str(data, "plane")
    if kind == "single":
        image = _require_str(data, "image")
        truth_positions = [_require_float(data, "position_mm")]
        images = [image]
        interval_um = None
    else:
        raw_images = data.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError(f"Manifest record {record_id!r} requires non-empty images")
        if not all(isinstance(item, str) and item for item in raw_images):
            raise ValueError(f"Manifest record {record_id!r} images must be strings")
        images = list(raw_images)
        truth_positions = _require_float_list(data, "positions_mm")
        if len(images) != len(truth_positions):
            raise ValueError(
                f"Manifest record {record_id!r} image/position length mismatch"
            )
        interval_um = _optional_int(data, "interval_um")
        if interval_um is None:
            raise ValueError(f"Manifest record {record_id!r} requires interval_um")

    known = {
        "id", "kind", "image", "images", "atlas", "plane", "position_mm",
        "positions_mm", "interval_um", "thickness_um", "metadata",
    }
    thickness_um = _optional_int(data, "thickness_um", 50)
    return TraceManifestRow(
        id=record_id,
        kind=cast(TraceKind, kind),
        images=images,
        atlas=atlas,
        plane=plane,
        truth_positions_mm=truth_positions,
        interval_um=interval_um,
        thickness_um=50 if thickness_um is None else thickness_um,
        metadata=_metadata(data, known),
    )


def load_manifest(path: str | Path) -> list[TraceManifestRow]:
    """Load a JSONL trace manifest."""

    rows: list[TraceManifestRow] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest line {line_no} is not an object")
        rows.append(parse_manifest_record(data))
    return rows


def estimate_cost_usd(
    usage: Mapping[str, int | float | str | bool], pricing: TracePricing
) -> float:
    """Estimate call cost, counting Gemini thinking tokens as output tokens."""

    prompt = float(usage.get("prompt_token_count") or 0)
    tool_prompt = float(usage.get("tool_use_prompt_token_count") or 0)
    cached = float(usage.get("cached_content_token_count") or 0)
    output = float(usage.get("candidates_token_count") or 0)
    thoughts = float(usage.get("thoughts_token_count") or 0)

    billable_input = max(0.0, prompt + tool_prompt - cached)
    cost = (
        billable_input * pricing.input_per_million
        + cached * pricing.cached_input_per_million
        + (output + thoughts) * pricing.output_per_million
    ) / 1_000_000.0
    return round(cost, 6)


def categorize_trace(
    *,
    truth_positions: list[float],
    submitted_positions: list[float] | None,
    fetched_positions: list[float],
    tolerance_mm: float,
    turn_count: int,
    median_turn_count: int,
    had_broad_restart: bool,
    submit_rejections: int,
    fallback: bool,
) -> dict[str, Any]:
    """Categorize a trace for SFT filtering and hard-negative accounting."""

    if fallback:
        return {"accepted": False, "label": "fallback"}
    if not submitted_positions:
        return {"accepted": False, "label": "no_submit"}
    if len(submitted_positions) != len(truth_positions):
        return {"accepted": False, "label": "length_mismatch"}

    errors = [abs(a - b) for a, b in zip(submitted_positions, truth_positions, strict=True)]
    max_error = max(errors) if errors else math.inf
    accepted = max_error <= tolerance_mm
    if not accepted:
        return {
            "accepted": False,
            "label": "out_of_tolerance",
            "max_error_mm": round(max_error, 4),
        }

    excursion = 0.0
    if fetched_positions and submitted_positions:
        centers = submitted_positions
        excursion = max(
            min(abs(fetched - center) for center in centers)
            for fetched in fetched_positions
        )
    hard = (
        turn_count > median_turn_count
        or had_broad_restart
        or submit_rejections > 0
    )
    return {
        "accepted": True,
        "label": "hard_negative_recovered" if hard else "clean_success",
        "max_error_mm": round(max_error, 4),
        "trajectory_excursion_mm": round(excursion, 4),
    }


def export_sft_messages(
    raw_events: list[dict[str, Any]],
    *,
    user_message: dict[str, Any] | None = None,
    include_rationale_summaries: bool = False,
) -> list[dict[str, Any]]:
    """Convert raw trace events to deployment-shaped SFT messages.

    Teacher thought summaries are omitted by default. Rationale exports include
    them as visible, compact ``Rationale summary: ...`` text rather than hidden
    chain-of-thought.
    """

    messages: list[dict[str, Any]] = []
    if user_message is not None:
        messages.append(user_message)
    for event in raw_events:
        role = event.get("role")
        if role == "user":
            messages.append({"role": "user", "parts": event.get("parts", [])})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant"}
            content_chunks: list[str] = []
            if include_rationale_summaries:
                summaries = event.get("teacher_thought_summaries")
                if isinstance(summaries, list):
                    for summary in summaries:
                        if isinstance(summary, str) and summary.strip():
                            content_chunks.append(
                                f"Rationale summary: {summary.strip()}"
                            )
            visible = event.get("visible_text")
            if isinstance(visible, str) and visible:
                content_chunks.append(visible)
            if content_chunks:
                msg["content"] = "\n\n".join(content_chunks)
            tool_calls = event.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                msg["tool_calls"] = tool_calls
            if "content" in msg or "tool_calls" in msg:
                messages.append(msg)
        elif role == "tool":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "name": event.get("name"),
                "response": event.get("response"),
            }
            artifacts = event.get("artifacts")
            if isinstance(artifacts, list) and artifacts:
                tool_msg["artifacts"] = artifacts
                content_parts: list[dict[str, Any]] = []
                response = event.get("response")
                description = (
                    response.get("description")
                    if isinstance(response, dict)
                    else None
                )
                if isinstance(description, str) and description:
                    content_parts.append({"type": "text", "text": description})
                content_parts.extend(cast(list[dict[str, Any]], artifacts))
                tool_msg["content"] = content_parts
            messages.append(tool_msg)
    return messages


def build_sft_user_message(
    row: TraceManifestRow, target_artifacts: list[dict[str, str]]
) -> dict[str, Any]:
    axis_label = {"coronal": "AP", "sagittal": "ML", "horizontal": "DV"}.get(
        row.plane, "position"
    )
    if row.kind == "single":
        text = (
            f"Determine this {row.plane} slice's {axis_label} position in "
            f"the {row.atlas} atlas."
        )
    else:
        text = (
            f"Determine each {row.plane} slice's {axis_label} position in "
            f"the {row.atlas} atlas. The images are ordered along the "
            f"{axis_label} axis."
        )
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, *target_artifacts],
    }


def _persist_target_images(images: list[Image.Image], run_dir: Path) -> list[dict[str, str]]:
    target_dir = run_dir / "target_images"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    for index, image in enumerate(images, start=1):
        path = target_dir / f"target_{index:03d}.jpg"
        image.convert("RGB").save(path, format="JPEG", quality=90)
        artifacts.append({
            "type": "image_path",
            "path": str(path.relative_to(run_dir)).replace("\\", "/"),
            "mime_type": "image/jpeg",
        })
    return artifacts


def assess_sft_quality(
    raw_events: list[dict[str, Any]],
    category: Mapping[str, Any],
    *,
    max_rationale_chars: int = 800,
) -> dict[str, Any]:
    missing_rationale_before_tool = 0
    overlong_rationale = 0
    assistant_tool_turns = 0
    for event in raw_events:
        if event.get("role") != "assistant":
            continue
        tool_calls = event.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        assistant_tool_turns += 1
        visible = event.get("visible_text")
        summaries = event.get("teacher_thought_summaries")
        rationale_chunks: list[str] = []
        if isinstance(visible, str) and visible.strip():
            rationale_chunks.append(visible)
        if isinstance(summaries, list):
            rationale_chunks.extend(
                item for item in summaries if isinstance(item, str) and item.strip()
            )
        if not rationale_chunks:
            missing_rationale_before_tool += 1
        if any(len(chunk) > max_rationale_chars for chunk in rationale_chunks):
            overlong_rationale += 1

    accepted_for_sft = (
        category.get("accepted") is True
        and missing_rationale_before_tool == 0
        and overlong_rationale == 0
    )
    return {
        "accepted_for_sft": accepted_for_sft,
        "assistant_tool_turns": assistant_tool_turns,
        "missing_rationale_before_tool": missing_rationale_before_tool,
        "overlong_rationale": overlong_rationale,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(payload)) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def _load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).copy() for path in paths]


def _submitted_positions(result: Any, kind: TraceKind) -> list[float] | None:
    if kind == "single":
        position = getattr(result, "position_mm", None)
        return [float(position)] if isinstance(position, (int, float)) else None
    positions = getattr(result, "positions", None)
    if not isinstance(positions, list):
        return None
    out: list[float] = []
    for item in positions:
        position = getattr(item, "position_mm", None)
        if not isinstance(position, (int, float)):
            return None
        out.append(float(position))
    return out


def _fetched_positions(events: list[dict[str, Any]]) -> list[float]:
    positions: list[float] = []
    for event in events:
        calls = event.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or call.get("name") != "fetch_atlas":
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            raw_positions = args.get("positions_mm")
            if not isinstance(raw_positions, list):
                continue
            for pos in raw_positions:
                if isinstance(pos, (int, float)):
                    positions.append(float(pos))
    return positions


def _had_broad_restart(events: list[dict[str, Any]]) -> bool:
    broad_calls = 0
    for event in events:
        calls = event.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or call.get("name") != "fetch_atlas":
                continue
            args = call.get("args")
            raw_positions = args.get("positions_mm") if isinstance(args, dict) else None
            if isinstance(raw_positions, list) and len(raw_positions) >= 3:
                span = max(float(p) for p in raw_positions) - min(float(p) for p in raw_positions)
                if span > 1.0:
                    broad_calls += 1
    return broad_calls > 1


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [item for item in results if item["category"].get("accepted") is True]
    labels: dict[str, int] = {}
    for item in results:
        label = str(item["category"].get("label", "unknown"))
        labels[label] = labels.get(label, 0) + 1
    total_cost = sum(float(item.get("estimated_cost_usd", 0.0)) for item in results)
    return {
        "total_runs": len(results),
        "accepted_runs": len(accepted),
        "labels": labels,
        "estimated_cost_usd": round(total_cost, 6),
    }


async def _collect_manifest_traces_async(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    model: str,
    thinking_level: str,
    media_resolution: str,
    max_iterations: int,
    include_thought_summaries: bool,
    sft_export: SftExportMode,
    persist_tool_images: bool,
    limit: int | None,
    kind_filter: TraceKind | None,
    resume: bool,
    single_runner: SingleRunner | None,
    group_runner: GroupRunner | None,
) -> list[dict[str, Any]]:
    from langslice_harness.harness.estimation.runner import (
        run_group_session,
        run_single_slice_session,
    )

    resolved_single = single_runner or run_single_slice_session
    resolved_group = group_runner or run_group_session
    rows = load_manifest(manifest_path)
    if kind_filter is not None:
        rows = [row for row in rows if row.kind == kind_filter]
    if limit is not None:
        rows = rows[:limit]

    root = Path(out_dir)
    results: list[dict[str, Any]] = []
    for row in rows:
        run_dir = root / "runs" / row.id
        raw_path = run_dir / "raw_trace.json"
        if resume and raw_path.exists():
            continue

        images = _load_images(row.images)
        target_artifacts = _persist_target_images(images, run_dir)
        user_message = build_sft_user_message(row, target_artifacts)
        recorder = AgentTraceRecorder(
            tool_artifact_dir=(run_dir / "tool_images") if persist_tool_images else None
        )
        common_kwargs = {
            "atlas_name": row.atlas,
            "plane": row.plane,
            "model": model,
            "media_resolution": media_resolution,
            "max_iterations": max_iterations,
            "thinking_level": thinking_level,
            "include_thought_summaries": include_thought_summaries,
            "trace_recorder": recorder,
        }
        if row.kind == "single":
            result = await resolved_single(image=images[0], **common_kwargs)
        else:
            result = await resolved_group(
                images=images,
                interval_mm=float(row.interval_um or 0) / 1000.0,
                thickness_um=row.thickness_um,
                **common_kwargs,
            )

        submitted = _submitted_positions(result, row.kind)
        fallback = submitted is None or any(
            "fell back" in str(getattr(item, "reasoning", "")).lower()
            for item in (getattr(result, "positions", None) or [result])
        )
        truth_canonical = canonicalize_positions(
            row.truth_positions_mm, row.atlas, row.plane
        )
        submitted_canonical = canonicalize_positions(submitted, row.atlas, row.plane)
        category = categorize_trace(
            truth_positions=truth_canonical,
            submitted_positions=submitted_canonical,
            fetched_positions=_fetched_positions(recorder.raw_events),
            tolerance_mm=plane_tolerance_mm(row.atlas, row.plane),
            turn_count=max(1, len(recorder.usage_by_call)),
            median_turn_count=6,
            had_broad_restart=_had_broad_restart(recorder.raw_events),
            submit_rejections=recorder.submit_rejections,
            fallback=fallback,
        )
        cost = estimate_cost_usd(
            recorder.usage_totals,
            GEMINI_31_PRO_PREVIEW_PRICING,
        )
        sft_quality = assess_sft_quality(recorder.raw_events, category)
        manifest_payload = {
            "id": row.id,
            "kind": row.kind,
            "images": row.images,
            "atlas": row.atlas,
            "plane": row.plane,
            "truth_positions_mm": row.truth_positions_mm,
            "interval_um": row.interval_um,
            "thickness_um": row.thickness_um,
            "metadata": row.metadata,
        }
        raw_payload = {
            "manifest": manifest_payload,
            "model": model,
            "thinking_level": thinking_level,
            "include_thought_summaries": include_thought_summaries,
            "events": recorder.raw_events,
            "usage_totals": recorder.usage_totals,
            "submitted_positions_mm": submitted,
            "category": category,
            "sft_quality": sft_quality,
            "estimated_cost_usd": cost,
        }
        _write_json(raw_path, raw_payload)
        deployment_payload = {
            "manifest": manifest_payload,
            "messages": export_sft_messages(
                recorder.raw_events,
                user_message=user_message,
                include_rationale_summaries=False,
            ),
        }
        rationale_payload = {
            "manifest": manifest_payload,
            "messages": export_sft_messages(
                recorder.raw_events,
                user_message=user_message,
                include_rationale_summaries=True,
            ),
        }
        _write_json(run_dir / "sft_trace.json", deployment_payload)
        if sft_export in {"deployment", "both"}:
            _write_json(run_dir / "sft_trace_deployment.json", deployment_payload)
        if sft_export in {"rationale", "both"}:
            _write_json(run_dir / "sft_trace_rationale.json", rationale_payload)
        _write_json(run_dir / "telemetry.json", {
            "usage_by_call": recorder.usage_by_call,
            "usage_totals": recorder.usage_totals,
            "estimated_cost_usd": cost,
            "category": category,
            "sft_quality": sft_quality,
        })
        results.append({
            "id": row.id,
            "kind": row.kind,
            "model": model,
            "submitted_positions_mm": submitted,
            "truth_positions_mm": row.truth_positions_mm,
            "category": category,
            "sft_quality": sft_quality,
            "estimated_cost_usd": cost,
        })

    _append_jsonl(root / "results.jsonl", results)
    summary = _summarize_results(results)
    _write_json(root / "trace_summary.json", summary)
    _write_json(root / "cost_summary.json", {
        "model": model,
        "pricing": GEMINI_31_PRO_PREVIEW_PRICING,
        "estimated_cost_usd": summary["estimated_cost_usd"],
    })
    return results


def collect_manifest_traces(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    model: str = "gemini-3.1-pro-preview",
    thinking_level: str = "MEDIUM",
    media_resolution: str = "medium",
    max_iterations: int = 20,
    include_thought_summaries: bool = True,
    sft_export: SftExportMode = "both",
    persist_tool_images: bool = True,
    limit: int | None = None,
    kind_filter: TraceKind | None = None,
    resume: bool = False,
    single_runner: SingleRunner | None = None,
    group_runner: GroupRunner | None = None,
) -> list[dict[str, Any]]:
    """Run a manifest-driven trace collection batch."""

    return asyncio.run(_collect_manifest_traces_async(
        manifest_path=manifest_path,
        out_dir=out_dir,
        model=model,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        max_iterations=max_iterations,
        include_thought_summaries=include_thought_summaries,
        sft_export=sft_export,
        persist_tool_images=persist_tool_images,
        limit=limit,
        kind_filter=kind_filter,
        resume=resume,
        single_runner=single_runner,
        group_runner=group_runner,
    ))
