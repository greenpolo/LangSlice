"""Shared helpers for the tool-door tests: a small stack, its toolbox, ADK's
tool context, and one-entry adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from tests.fakes import SlabAtlas

#: The atlas of :func:`stack`: 20 slices, 1 mm each (positions 0-19 mm).
SLAB = SlabAtlas()


def tool_named(box, name: str):
    """The tool called *name* in a built toolbox."""
    return next(tool for tool in box.tools if tool.__name__ == name)


class Actions:
    escalate = False


class ToolContext:
    """What ADK hands a tool as ``tool_context``: the call id and the actions."""

    def __init__(self, function_call_id: str | None = None) -> None:
        self.actions = Actions()
        self.function_call_id = function_call_id


def stack(folder: Path, n: int = 5, *, placed: bool = False, atlas: Any = None,
          **spec_kwargs: Any):
    """``(state, ctx, spec)``: *n* flat gray sections ``s0.png``... ingested on
    *atlas* (the slab by default), with no starting positions; *placed* puts
    them 0.5 mm apart from 2 mm in file order."""
    for index in range(n):
        Image.fromarray(
            np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)
        ).save(folder / f"s{index}.png")
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   **spec_kwargs)
    the_atlas = SLAB if atlas is None else atlas
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: the_atlas)
    state = ingest(spec, ctx)
    if placed:
        for index, record in enumerate(state.in_order()):
            record.position_mm = 2.0 + 0.5 * index
    return state, ctx, spec


def box(folder: Path, **kwargs: Any):
    """``(state, ctx, toolbox)`` over :func:`stack`."""
    state, ctx, spec = stack(folder, **kwargs)
    return state, ctx, build_tools(state, ctx, spec)


def submit(toolbox, **kwargs: Any) -> dict[str, Any]:
    """Call ``submit`` with an empty summary, notes and breaks unless given."""
    args: dict[str, Any] = {"summary": "done", "notes": [], "interval_breaks": []}
    args.update(kwargs)
    return tool_named(toolbox, "submit")(**args, tool_context=ToolContext())


def single_transform(tool):
    """Call ``interactive_transform`` with one section and return its row.

    Positional values are rotation_deg, scale_x, scale_y, translate_x_mm and
    translate_y_mm; any other entry key goes as a keyword. The row carries
    the call's pictures (``pictures`` and the images); a refusal of the whole
    call comes back as it is.
    """
    def transform(slice_id, *args, view: bool = True, **kwargs):
        fields = ("rotation_deg", "scale_x", "scale_y", "translate_x_mm", "translate_y_mm")
        entry = {"id": slice_id, **dict(zip(fields, args, strict=False)), **kwargs}
        result = tool([entry], view=view)
        if "results" not in result:
            return result
        row = dict(result["results"][0])
        row["pictures"] = result.get("pictures", [])
        row[TOOL_MEDIA_PARTS_KEY] = result.get(TOOL_MEDIA_PARTS_KEY, [])
        if "changed" in result:
            row["changed"] = result["changed"]
        return row
    return transform
