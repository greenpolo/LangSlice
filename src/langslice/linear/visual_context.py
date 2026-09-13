"""Completion-based image retirement within the original conversation.

The registry owns attachment identities, not scientific state or model memory.
It never summarizes or edits historical text. Only an explicit acceptance of
previously delivered, current geometry retires superseded image attachments.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.genai import types

from langslice.adk import MEDIA_LAYOUT_ATTR, TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.linear.state import SliceState, StackState

# Tool-internal handoff, removed before ADK sees the result.
EVIDENCE_KEY = "_langslice_visual_evidence"
# Private model_copy attribute: deliberately absent from FunctionResponse JSON.


@dataclass
class VisualEvidence:
    """One attachment, including its ownership and the state actually drawn."""

    slice_id: str | None = None
    stage: str = "shared"
    geometry: str = ""
    eligible: bool = False
    placement_key: tuple[str, float, bool, int, float, float] | None = None
    delivered: bool = False
    retired: bool = False
    # Required companion slots in the same response (e.g. the corrected
    # section accompanying an atlas candidate). They form one accepted view.
    dependencies: tuple[int, ...] = ()


class VisualContext:
    """A per-session registry and callable ADK request filter."""

    def __init__(
        self,
        state: StackState,
        seen_placements: set[tuple[str, float, bool, int, float, float]],
        *,
        final_stage: str,
    ) -> None:
        self.state = state
        self.seen_placements = seen_placements
        self.final_stage = final_stage
        self.media: dict[str, list[VisualEvidence]] = {}
        self.accepted: dict[tuple[str, str], tuple[str, int]] = {}
        self.trace: Callable[[dict[str, Any]], None] | None = None
        self._events: list[dict[str, Any]] = []

    def geometry(self, record: SliceState, stage: str, position: float | None = None) -> str:
        """Match scientific geometry, ignoring notes and fitting provenance."""
        value: list[Any] = [
            record.id, record.position_mm if position is None else position,
            record.flip, record.rotation_deg, self.state.pitch_deg, self.state.yaw_deg,
        ]
        if stage == "transform":
            value.append((record.transform or {}).get("params"))
        return json.dumps(value, separators=(",", ":"))

    def evidence(
        self, record: SliceState, stage: str, *, eligible: bool,
        position: float | None = None,
    ) -> VisualEvidence:
        placement = None
        if stage == "position" and eligible:
            drawn_position = record.position_mm if position is None else position
            if drawn_position is None:
                raise ValueError("An eligible placement view must have a position")
            placement = (
                record.id, float(drawn_position),
                record.flip, record.rotation_deg, self.state.pitch_deg, self.state.yaw_deg,
            )
        return VisualEvidence(
            record.id, stage, self.geometry(record, stage, position), eligible, placement,
        )

    def register(self, result: dict[str, Any]) -> None:
        """Assign one response identity and exact slots before ADK lifts media."""
        evidence = result.pop(EVIDENCE_KEY, None)
        raw_parts = result.get(TOOL_MEDIA_PARTS_KEY)
        if not isinstance(raw_parts, list):
            return
        parts = [part for part in raw_parts if getattr(part, "inline_data", None) is not None]
        if not parts:
            return
        if evidence is None:
            evidence = [VisualEvidence() for _ in parts]
        if len(evidence) != len(parts):
            raise ValueError("Visual ownership must match every rendered attachment")
        if any(index < 0 or index >= len(parts)
               for item in evidence for index in item.dependencies):
            raise ValueError("Visual dependency must reference an attachment in this response")
        delivery_id = uuid.uuid4().hex
        result[TOOL_MEDIA_DELIVERY_ID_KEY] = delivery_id
        self.media[delivery_id] = evidence
        self._events.append({
            "kind": "image_registered",
            "attachments": [{
                "id": f"{delivery_id}:{index}", "slice_id": item.slice_id,
                "stage": item.stage, "geometry": item.geometry,
                "eligible": item.eligible,
                "dependencies": [f"{delivery_id}:{slot}" for slot in item.dependencies],
            } for index, item in enumerate(evidence)],
        })
        self.flush_events()

    def _current(self, identity: tuple[str, int], stage: str) -> bool:
        item = self.media[identity[0]][identity[1]]
        record = self.state.by_id(item.slice_id or "")
        return bool(
            record is not None and not item.retired and item.delivered and item.eligible
            and item.stage == stage and item.geometry == self.geometry(record, stage)
            and all(self.media[identity[0]][index].delivered
                    and not self.media[identity[0]][index].retired
                    for index in item.dependencies)
        )

    def candidates(
        self, slice_ids: list[str], stage: str,
    ) -> tuple[dict[str, tuple[str, int]], list[str]]:
        chosen: dict[str, tuple[str, int]] = {}
        for delivery_id, entries in self.media.items():
            for index, item in enumerate(entries):
                identity = (delivery_id, index)
                if item.slice_id in slice_ids and self._current(identity, stage):
                    chosen[item.slice_id] = identity
        return chosen, [slice_id for slice_id in slice_ids if slice_id not in chosen]

    def accept(self, slice_ids: list[str], stage: str, *, final: bool = False) -> dict[str, Any]:
        """Atomically accept inspected current views and retire their alternatives."""
        chosen, missing = self.candidates(slice_ids, stage)
        if missing:
            return {
                "status": "refused", "error": "FINAL_VIEWS_NOT_SEEN", "stage": stage,
                "slice_ids": missing,
                "detail": "A full current comparison or transform overlay must reach "
                "an earlier model request before acceptance.",
            }
        self.accepted.update({(stage, key): value for key, value in chosen.items()})
        all_ids = [record.id for record in self.state.slices]
        complete = all(
            (identity := self.accepted.get((stage, slice_id))) is not None
            and self._current(identity, stage) for slice_id in all_ids
        )
        protected = {
            identity for (accepted_stage, _), identity in self.accepted.items()
            if accepted_stage == stage and self._current(identity, stage)
        }
        protected.update({
            (delivery_id, dependency)
            for delivery_id, index in list(protected)
            for dependency in self.media[delivery_id][index].dependencies
        })
        retired: list[str] = []
        for delivery_id, entries in self.media.items():
            for index, item in enumerate(entries):
                identity = (delivery_id, index)
                local = item.slice_id in chosen and (
                    item.stage in {stage, "reference"}
                    or stage == "transform" and item.stage == "position"
                )
                # Generic shared evidence may inform either kind of work.
                # Position completion cannot retire references for open transforms.
                shared = complete and stage == self.final_stage and item.stage == "shared"
                if (item.delivered and not item.retired and identity not in protected
                        and (local or shared or final)):
                    item.retired = True
                    retired.append(f"{delivery_id}:{index}")
        event = {
            "kind": "image_retirement", "stage": stage, "slice_ids": list(chosen),
            "retired": retired,
            "accepted": {key: f"{value[0]}:{value[1]}" for key, value in chosen.items()},
            "protected_images": sorted(f"{key}:{index}" for key, index in protected),
            "final": final,
        }
        self._events.append(event)
        self.flush_events()
        return {
            "status": "ok", "stage": stage, "accepted": list(chosen),
            "retired_images": len(retired),
        }

    def mark_all_delivered(self) -> None:
        """Explicit model boundary for direct host/unit-test tool invocation."""
        for entries in self.media.values():
            for item in entries:
                if not item.retired:
                    item.delivered = True
                    if item.placement_key is not None:
                        self.seen_placements.add(item.placement_key)

    def flush_events(self) -> None:
        if self.trace is not None:
            for event in self._events:
                self.trace(event)
        self._events.clear()

    def __call__(self, contents: list[types.Content]) -> list[types.Content]:
        """Copy only changed response containers; preserve all original text."""
        out = list(contents)
        present: list[str] = []
        for ci, content in enumerate(contents):
            copied = list(content.parts or [])
            changed = False
            for pi, part in enumerate(content.parts or []):
                response = part.function_response
                if response is None or not isinstance(response.response, dict):
                    continue
                delivery_id = response.response.get(TOOL_MEDIA_DELIVERY_ID_KEY)
                if not isinstance(delivery_id, str):
                    continue
                entries = self.media.get(delivery_id, [])
                if not entries or not response.parts:
                    continue
                # The ADK history is untouched, so every request normally starts
                # with all original slots. Also support filtering an already
                # filtered copy without renumbering its surviving attachments.
                original = getattr(response, MEDIA_LAYOUT_ATTR, None)
                indexes = original[1] if original is not None else list(range(len(response.parts)))
                keep = []
                kept_indexes = []
                for index, attachment in zip(indexes, response.parts, strict=True):
                    item = entries[index]
                    if item.retired:
                        continue
                    item.delivered = True
                    if item.placement_key is not None:
                        self.seen_placements.add(item.placement_key)
                    present.append(f"{delivery_id}:{index}")
                    keep.append(attachment)
                    kept_indexes.append(index)
                if len(keep) != len(response.parts):
                    replacement = response.model_copy(update={
                        "parts": keep or None,
                        MEDIA_LAYOUT_ATTR: (len(entries), kept_indexes),
                    })
                    copied[pi] = part.model_copy(update={"function_response": replacement})
                    changed = True
            if changed:
                out[ci] = content.model_copy(update={"parts": copied})
        self.flush_events()
        if self.trace is not None:
            self.trace({"kind": "image_context", "present": present})
        return out
