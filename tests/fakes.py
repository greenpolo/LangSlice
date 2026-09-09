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

    Minimal generic atlas fixture for the linear tests:
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


_QUOTA_CALLS = [0]


class _StackLlm(BaseLlm):
    """A fake BaseLlm that drives one linear stack session to ``submit``.

    ``positions`` (id -> mm) is written with ``set_positions`` on the first
    turn — ``submit`` refuses a stack with any section still unplaced — and the
    submission follows on the next.
    """

    positions: dict[str, float] | None = None
    #: Input tokens reported per call, so budget handling can be tested.
    input_tokens_per_call: int = 0
    #: Usage-window percent the fake reports as used, stepping this much per
    #: call from 40, the way the OAuth lane's quota headers do.
    quota_percent_per_call: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        available = set(llm_request.tools_dict or {})
        if _has_function_response(llm_request, "submit"):
            # The job is done; anything after this is the debrief question.
            part = types.Part.from_text(text="Debrief: nothing was missing.")
        elif (
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
        elif "submit" in available:
            part = types.Part.from_function_call(
                name="submit",
                args={"summary": "Placed the stack.", "notes": [], "interval_breaks": []},
            )
        else:
            part = types.Part.from_text(text="Nothing to submit.")
        # ADK resolves a fresh instance per turn, so the call count lives here.
        _QUOTA_CALLS[0] += 1
        quota = (
            {"quota": {"primary_used_percent": str(40 + _QUOTA_CALLS[0] * self.quota_percent_per_call)}}
            if self.quota_percent_per_call
            else None
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=False,
            turn_complete=True,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=self.input_tokens_per_call,
                candidates_token_count=1,
            ),
            custom_metadata=quota,
        )


def install_fake_adk_model_stack(
    monkeypatch: Any,
    positions: dict[str, float] | None = None,
    input_tokens_per_call: int = 0,
    quota_percent_per_call: int = 0,
) -> None:
    """Patch LLMRegistry.new_llm so a linear session submits at once."""
    from google.adk.models.registry import LLMRegistry

    _QUOTA_CALLS[0] = 0

    def _fake_new_llm(model: str) -> BaseLlm:
        return _StackLlm(
            model=model,
            positions=positions,
            input_tokens_per_call=input_tokens_per_call,
            quota_percent_per_call=quota_percent_per_call,
        )

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(_fake_new_llm))


def _has_function_response(llm_request: LlmRequest, name: str) -> bool:
    """Whether a response from tool *name* already sits in the request history."""
    for content in llm_request.contents or []:
        for part in getattr(content, "parts", None) or []:
            response = getattr(part, "function_response", None)
            if response is not None and getattr(response, "name", None) == name:
                return True
    return False


def _count_function_responses(llm_request: LlmRequest) -> int:
    """Count how many function-response parts appear across request contents."""
    count = 0
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if getattr(part, "function_response", None) is not None:
                count += 1
    return count
