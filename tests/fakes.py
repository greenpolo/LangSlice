"""Test doubles: ADK LlmAgent model invocations, plus a synthetic atlas.

## Synthetic atlas

``EllipseAtlas``/``ellipse_section`` are the smallest pair of shapes the
silhouette affine can actually register — one ellipse of tissue in a 20-slice,
1 mm-per-voxel volume, and a section holding another. Small enough to build in
a test, real enough that the moments fit has something to fit.

## ADK model doubles

The ADK `BaseLlm` abstraction (``google.adk.models.BaseLlm``) exposes a single
entry point for a model turn: ``generate_content_async(llm_request, stream)``.
`LlmAgent.canonical_model` returns either an explicit ``BaseLlm`` passed via
``model=...`` or delegates to ``LLMRegistry.new_llm(model_name)`` when a string
is provided. Monkeypatching ``LLMRegistry.new_llm`` lets us swap in a scripted
fake without touching the runner or agent builder.

The fake below emits a fixed sequence of turns. It reads the `contents` of the
incoming `LlmRequest` to figure out what step it is on (it counts how many
function-response parts it has received so far) and yields the corresponding
scripted response. Each response is a single ``LlmResponse`` with
``partial=False`` and a ``types.Content`` containing one function-call part.

This matches the non-streaming contract documented in ``BaseLlm``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import cv2
import numpy as np
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image


class EllipseAtlas:
    """A 20-slice atlas, 1 mm per voxel, whose tissue is one ellipse.

    1 mm voxels keep the volume tiny while leaving positions in the same
    0-19 mm ballpark the real coronal atlases use.
    """

    atlas_name = "fake_ellipse_1mm"
    orientation = "asr"
    resolution = (1000.0, 1000.0, 1000.0)
    metadata = {"species": "mouse"}

    #: Id of the one structure the ellipse is made of.
    BODY_ID = 2

    def __init__(
        self, *, height: int = 96, width: int = 128, axes: tuple[int, int] = (46, 30)
    ):
        plane = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            plane, (width // 2, height // 2), axes, 0, 0, 360, self.BODY_ID, -1
        )
        self.annotation = np.repeat(plane[None, :, :], 20, axis=0)
        self.template = (self.annotation > 0).astype(np.uint8) * 200


class SlabAtlas:
    """A 20-slice, 1 mm-per-voxel atlas with a ``root`` tissue shell.

    Minimal generic atlas fixture for the whole-brain positioning tests:
    enough shape (annotation, template, resolution, orientation) for
    ``ingest`` to compute a position range and render a contact sheet.
    """

    atlas_name = "fake_slab_1mm"
    orientation = "asr"
    resolution = (1000.0, 1000.0, 1000.0)
    metadata = {"species": "mouse"}

    def __init__(self, *, n_slices: int = 20, height: int = 12, width: int = 16):
        annotation = np.zeros((n_slices, height, width), dtype=np.int32)
        annotation[:, 2 : height - 2, 2 : width - 2] = 1  # tissue on every slice
        self.annotation = annotation
        self.template = (annotation > 0).astype(np.uint8) * 200


def ellipse_section(
    size: tuple[int, int] = (200, 160),
    axes: tuple[int, int] = (70, 40),
    angle: float = 0.0,
) -> Image.Image:
    """Dark tissue ellipse on a light field, like a scanned section."""
    width, height = size
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.ellipse(
        canvas, (width // 2, height // 2), axes, angle, 0, 360, (30, 30, 30), -1
    )
    return Image.fromarray(canvas, mode="RGB")


class _ScriptedSubmitLlm(BaseLlm):
    """A fake BaseLlm that scripts broad sweep -> narrow sweep -> submit_estimate.

    The step is inferred from how many function-response parts are already
    present in the request. This avoids needing instance state on the fake and
    keeps behaviour deterministic across retries.
    """

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream  # unused; we always emit one response
        n_responses = _count_function_responses(llm_request)

        if n_responses == 0:
            # Turn 1: broad sweep
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [2.0, 4.0, 6.0, 8.0, 10.0]},
            )
        elif n_responses == 1:
            # Turn 2: narrow sweep around a plausible candidate
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [5.6, 5.8, 6.0, 6.2, 6.4]},
            )
        else:
            # Turn 3+: submit. The validator may reject the first few attempts
            # if gating is strict, but will relax after enough attempts.
            call = types.Part.from_function_call(
                name="submit_estimate",
                args={
                    "position_mm": 6.0,
                    "reasoning": "Broad sweep placed slice near 6mm; narrow sweep confirmed.",
                },
            )

        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
            partial=False,
            turn_complete=True,
        )


_CLEAN_STACK_SUBMISSIONS: dict[str, dict[str, Any]] = {
    "submit_survey": {
        "axis_directions": {"ap": "anterior_to_posterior"},
        "interval_breaks": [],
        "notes": [],
        "clean": True,
        "summary": "Stack is consistent.",
    },
    "submit_positions": {
        "interval_breaks": [],
        "notes": [],
        "summary": "Placed the stack from two key sections.",
    },
    "submit_review": {
        "approved": True,
        "notes": [],
        "summary": "Stack is consistent end to end.",
    },
}


class _CleanStackLlm(BaseLlm):
    """A fake BaseLlm that submits a clean result for any whole-brain step.

    Which step it is in is read off the tools the request declares, so one
    fake drives the survey and positioning agents alike.

    ``positions`` (id -> mm) is the positioning step's script: the stack now
    reaches that step unplaced, so the fake writes those positions with
    ``set_positions`` on its first turn and submits on the next.
    """

    positions: dict[str, float] | None = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        available = set(llm_request.tools_dict or {})
        if (
            self.positions
            and "set_positions" in available
            and _count_function_responses(llm_request) == 0
        ):
            part = types.Part.from_function_call(
                name="set_positions",
                args={
                    "entries": [
                        {"id": slice_id, "position_mm": position}
                        for slice_id, position in self.positions.items()
                    ]
                },
            )
        else:
            name = next(
                (tool for tool in _CLEAN_STACK_SUBMISSIONS if tool in available), None
            )
            args = dict(_CLEAN_STACK_SUBMISSIONS[name]) if name is not None else {}
            part = (
                types.Part.from_function_call(name=name, args=args)
                if name is not None
                else types.Part.from_text(text="Nothing to submit.")
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=False,
            turn_complete=True,
        )


def install_fake_adk_model_clean_stack(
    monkeypatch: Any, positions: dict[str, float] | None = None
) -> None:
    """Patch LLMRegistry.new_llm so whole-brain agent steps submit at once.

    Pass *positions* (slice id -> mm) when the run reaches the positioning
    step: it arrives unplaced, and ``submit_positions`` refuses a stack with
    any section still missing a position.
    """
    from google.adk.models.registry import LLMRegistry

    def _fake_new_llm(model: str) -> BaseLlm:
        return _CleanStackLlm(model=model, positions=positions)

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(_fake_new_llm))


def _count_function_responses(llm_request: LlmRequest) -> int:
    """Count how many function-response parts appear across request contents."""
    count = 0
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if getattr(part, "function_response", None) is not None:
                count += 1
    return count


def install_fake_adk_model_scripted_submit(monkeypatch: Any) -> None:
    """Patch LLMRegistry.new_llm so any ``model=<string>`` resolves to the fake.

    The resulting agent will, over the course of a session:
      1. call fetch_atlas with a broad sweep
      2. call fetch_atlas with a narrow sweep
      3. call submit_estimate
    """
    from google.adk.models.registry import LLMRegistry

    def _fake_new_llm(model: str) -> BaseLlm:
        return _ScriptedSubmitLlm(model=model)

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(_fake_new_llm))
