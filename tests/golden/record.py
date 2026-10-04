"""Golden snapshots of every linear tool: the recorder.

Phase 0 of the layered-core refactor (``_local/notes/20261001_layered_core_plan.md``):
a fixed, scripted sequence of tool calls on a small synthetic stack, recorded
so every later phase can show that the pictures stay pixel-identical and the
data and text stay equal.

What is recorded, per call: every picture (decoded and saved as PNG), the
structured data the tool returned (JSON), and the text the model receives.
Four doors are covered:

- ``tools``: the tools ``build_tools`` returns, called directly (the closures
  the later phases move). Two toolboxes: the full spec (every task, every
  optional tool, image model stubbed, picture size ``low``) and an ``auto``
  picture-size spec with the deformable engine fixed and no image model.
- ``engine``: ``langslice.linear.engine.run`` with a fake ADK model that
  records the first model request (the job statement, the opening strips and
  the status table, the tool declarations) and stops the session.
- ``mcp``: the MCP server driven by an in-memory client (``start_job``,
  ``show_stack`` pages and a few tools), as Claude hosts see it.
- ``gated``: the look-before-commit gates and the model-delivery
  bookkeeping on a ``position.gated`` toolbox: a write refused until the
  placement was viewed, a picture returned in the same model round and
  suppressed once delivered (by call id, and for direct calls by
  ``begin_model_call``), submit refused until ``view_stack``.
- ``resume``: a checkpoint -> reload -> continue round trip through the MCP
  door: one server writes, a second server on the same folder resumes from
  the checkpoint and carries on (including ``undo``/``redo`` across it).
- the final stack state of the main toolbox and of the resumed job.
- ``declarations``: what each door declares per tool (ADK function
  declarations of three toolboxes, MCP tool lists with their schemas),
  recorded last.
- ``job_folders``: each door's job folder (``<images>/langslice``) after
  the run: its file list, the views index, ``job.json``, a hash of every
  saved picture's label and border layers (decoded pixels) and of its
  ``view.json``, whether every saved ``view.jpg`` is byte for byte the
  JPEG the door sent, in order, and the derived public files
  (``registration.json``, each section's maps, the exports: decoded pixels
  and normalised JSON hashed, ``labels.csv`` as text).

Everything goes through public entry points (``build_tools``, ``engine.run``,
``mcp_server.server.build_server``). The few internals touched are listed in
:data:`PATCHES`; each is checked to exist, so a phase that moves one fails
here loudly instead of silently recording something else. The image model is
injected (:func:`stub_image_model`, given to ``build_tools``).

Regenerate deliberately::

    .venv/bin/python -m tests.golden.record            # rewrites tests/golden/linear_tools
    .venv/bin/python -m tests.golden.record --out DIR  # writes elsewhere (what the test does)

or run the test with ``LANGSLICE_UPDATE_GOLDEN=1``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

GOLDEN_DIR = Path(__file__).resolve().parent / "linear_tools"

#: Stand-in for the temporary folder every run works in.
ROOT_TOKEN = "<ROOT>"
#: Stand-in for a hex digest.
HEX_TOKEN = "<hex>"
#: Hex runs this long or longer are digests. Normalised because the image
#: correction's geometry fingerprint hashes the section file's absolute path
#: and mtime (``core.handoff.correction_fingerprint``), and the trace
#: call key, the deformable cache keys and the record directories derive from
#: it; none of them are stable across temporary folders.
HEX_RUN = re.compile(r"[0-9a-f]{16,}")
#: Keys whose values are wall-clock durations, normalised to this token.
SECONDS_TOKEN = "<seconds>"
SECONDS_KEYS = frozenset({"runtime_s", "elapsed_s", "seconds", "wall_s"})
#: Fixed mtime for the section files (2026-01-01 UTC), so nothing derived from
#: it varies even before normalisation.
FIXED_MTIME = 1767225600

ID0, ID1, ID2 = "s0.png", "s1.png", "s2.png"
#: The synthetic sections are drawn at 25 um/px (tests/deformable_synthetic)
#: and shrunk by this factor.
SECTION_SHRINK = 0.8
PIXEL_SIZE_UM = 25.0 / SECTION_SHRINK


# --- the stack ----------------------------------------------------------------


def write_sections(folder: Path) -> None:
    """Three 340 x 260 sections of the synthetic atlas, deterministic.

    s0: grey brightfield under a smooth residual field. s1: a colour section
    (red = the stain, green = a blurred copy, blue = flat) so the channel
    options have three channels to show. s2: tissue missing on the right
    (the section marked damaged later).
    """
    import cv2

    from tests.deformable_synthetic import (
        SECTION_SIZE,
        SMOOTH_FIELD,
        SyntheticAtlas,
        bump_field,
        render_section,
    )

    def smooth(image: Image.Image) -> np.ndarray:
        # The helper's pixel noise only inflates the PNGs; the template's
        # texture (a ~25 px period) survives this blur. Shrunk by
        # SECTION_SHRINK (the pixel size grows to match, PIXEL_SIZE_UM) so
        # every picture, and the goldens, are smaller.
        plane = cv2.GaussianBlur(np.asarray(image)[..., 0], (0, 0), 1.2)
        size = (round(plane.shape[1] * SECTION_SHRINK), round(plane.shape[0] * SECTION_SHRINK))
        return cv2.resize(plane, size, interpolation=cv2.INTER_AREA)

    def gray3(plane: np.ndarray) -> Image.Image:
        return Image.fromarray(np.stack([plane] * 3, axis=-1))

    atlas = SyntheticAtlas()
    folder.mkdir(parents=True, exist_ok=True)
    s0, _ = render_section(atlas, SMOOTH_FIELD(), seed=0)
    gray3(smooth(s0)).save(folder / ID0)
    gray, _ = render_section(atlas, bump_field([(170, 110, 50, -0.06, 0.05)]), seed=1)
    red = smooth(gray)
    green = cv2.GaussianBlur(255 - red, (0, 0), 2.0)
    blue = np.full_like(red, 30)
    Image.fromarray(np.stack([red, green, blue], axis=-1)).save(folder / ID1)
    width, height = SECTION_SIZE
    remove = np.zeros((height, width), dtype=bool)  # on the helper's grid
    remove[:, int(width * 0.68):] = True
    s2, _ = render_section(atlas, bump_field([(120, 140, 40, 0.04, -0.04)]), remove=remove,
                           seed=2)
    gray3(smooth(s2)).save(folder / ID2)
    for name in (ID0, ID1, ID2):
        os.utime(folder / name, (FIXED_MTIME, FIXED_MTIME))


def atlas_loader() -> Callable[[str], Any]:
    from tests.deformable_synthetic import SyntheticAtlas

    atlas = SyntheticAtlas()
    return lambda _name: atlas


# --- internals touched (each checked to exist) ----------------------------------


#: Module attributes set for the recording: fits run in this process (no
#: spawn pool, whose start-up costs seconds per call) at the coarse detail
#: level, as tests/test_linear_fit_deformable.py does. Same code path,
#: smaller pyramid.
PATCHES: tuple[tuple[str, str, Any], ...] = (
    ("langslice.linear.deformation", "USE_PROCESS_POOL", False),
    ("langslice.linear.deformation", "DETAIL_LEVEL", "coarse"),
)


def apply_patches() -> None:
    import importlib

    for module_name, attribute, value in PATCHES:
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            raise RuntimeError(f"golden recorder: {module_name}.{attribute} no longer exists; "
                               "update tests/golden/record.py PATCHES")
        setattr(module, attribute, value)


def stub_image_model(spec: Any) -> Any:
    """The image model the full toolbox is given: the run's, its network call replaced.

    ``build_tools`` takes the image model as an argument (phase 4); this is
    the spec's provider and model as :func:`langslice.providers.registry.resolve_image_model`
    resolves them, with a ``call`` that answers the edit with the rough border
    overlay ``registration_tool.start_correction`` prepared (the borders as
    drawn), so line extraction, the artifacts and the traced fit sections
    downstream all run on real data. No network.
    """
    import dataclasses

    from langslice.nonlinear.types import GeneratedSegmentation
    from langslice.providers.registry import resolve_image_model

    def generate(request: Any) -> GeneratedSegmentation:
        images = [request.slice_image, *request.reference_images]
        # The rough overlay is the one with drawn borders (more colour).
        def colour(image: Image.Image) -> float:
            pixels = np.asarray(image.convert("RGB")).astype(np.int16)
            return float(np.abs(pixels[..., 0] - pixels[..., 2]).mean())

        drawn = max(images, key=colour)
        return GeneratedSegmentation(image=drawn.convert("RGB"), provider=request.provider,
                                     model=str(request.model), route="golden-stub")

    resolved = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)
    return dataclasses.replace(resolved, call=generate)


# --- normalisation ----------------------------------------------------------------


class Normaliser:
    """Turns a run's volatile values into stable tokens (see module constants)."""

    def __init__(self, root: Path) -> None:
        roots = {str(root), os.path.realpath(root)}
        self.roots = sorted(roots, key=len, reverse=True)

    def text(self, value: str) -> str:
        for root in self.roots:
            value = value.replace(root, ROOT_TOKEN)
        return HEX_RUN.sub(HEX_TOKEN, value)

    def data(self, value: Any, key: str = "") -> Any:
        if key in SECONDS_KEYS and isinstance(value, (int, float)):
            return SECONDS_TOKEN
        if isinstance(value, dict):
            return {self.text(str(k)): self.data(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.data(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Path):
            return self.text(str(value))
        if isinstance(value, np.generic):
            return self.data(value.item(), key)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self.text(f"<{type(value).__name__}> {value}")


def canonical(value: Any) -> str:
    """The one JSON spelling goldens are written and compared in."""
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.copy()


# --- the recorder ------------------------------------------------------------------


class Recorder:
    """Writes one JSON per call plus its pictures as PNG into *out*."""

    def __init__(self, out: Path, root: Path) -> None:
        self.out = out
        self.norm = Normaliser(root)
        self.count = 0
        self.called: set[str] = set()
        self.started = time.perf_counter()
        #: Per door, the JPEG bytes of every picture it sent, in order.
        self.sent: dict[str, list[bytes]] = {}

    def _name(self, door: str, label: str) -> str:
        self.count += 1
        return f"{self.count:03d}_{door}_{label}"

    def _images(self, stem: str, images: list[Image.Image]) -> list[dict[str, Any]]:
        listed = []
        for index, image in enumerate(images):
            file = f"{stem}.{index}.png"
            image.save(self.out / file, format="PNG", optimize=True)
            listed.append({"file": file, "size": list(image.size), "mode": image.mode})
        return listed

    def write(self, stem: str, record: dict[str, Any]) -> None:
        (self.out / f"{stem}.json").write_text(canonical(record), encoding="utf-8")

    def tool(self, door: str, tools: dict[str, Any], name: str, *args: Any,
             **kwargs: Any) -> dict[str, Any]:
        """Call one toolbox tool and record what the ADK agent receives from it.

        The tools return plain pictures; the ADK door packages them as
        message parts (:func:`langslice.adk.media.package_result`), which
        are what is recorded.
        """
        from google.genai import types

        from langslice.adk import TOOL_MEDIA_PARTS_KEY
        from langslice.adk.media import package_result

        result = tools[name](*args, **kwargs)
        self.called.add(name)
        stem = self._name(door, name)
        packaged = package_result(result)
        body = dict(packaged) if isinstance(packaged, dict) else {"<result>": packaged}
        media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
        images: list[Image.Image] = []
        texts: list[str] = []
        if isinstance(media, list):
            for part in media:
                if isinstance(part, types.Part) and part.inline_data is not None:
                    self.sent.setdefault(door, []).append(part.inline_data.data or b"")
                    images.append(decode_image(part.inline_data.data or b""))
                elif isinstance(part, types.Part) and part.text is not None:
                    texts.append(self.norm.text(part.text))
                else:
                    texts.append(self.norm.text(f"<unexpected media> {part!r}"))
        elif media is not None:
            body[TOOL_MEDIA_PARTS_KEY] = media
        self.write(stem, {
            "door": door, "tool": name,
            "args": self.norm.data(list(args)), "kwargs": self.norm.data(kwargs),
            # The tool's reply minus its pictures: what the model reads as the
            # function response.
            "data": self.norm.data(body),
            "media_texts": texts,
            "images": self._images(stem, images),
        })
        return result

    def blocks(self, door: str, label: str, call: dict[str, Any], blocks: list[Any]) -> None:
        """Record MCP content blocks: each text verbatim, each image as pixels."""
        from mcp.types import ImageContent, TextContent

        stem = self._name(door, label)
        content: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for block in blocks:
            if isinstance(block, TextContent):
                content.append({"text": self.norm.text(block.text)})
            elif isinstance(block, ImageContent):
                content.append({"image": len(images), "mime": block.mimeType})
                self.sent.setdefault(door, []).append(base64.b64decode(block.data))
                images.append(decode_image(base64.b64decode(block.data)))
            else:
                content.append({"other": self.norm.text(repr(block))})
        self.write(stem, {"door": door, "call": self.norm.data(call), "content": content,
                          "images": self._images(stem, images)})

    def raw(self, door: str, label: str, record: dict[str, Any],
            images: list[Image.Image] = ()) -> None:  # type: ignore[assignment]
        stem = self._name(door, label)
        self.write(stem, {"door": door, **self.norm.data(record),
                          "images": self._images(stem, list(images))})


# --- door 1: the toolbox, full spec --------------------------------------------------


def full_spec(folder: Path) -> Any:
    from langslice.linear.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec

    return JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["reorder", "position", "transform", "nonlinear"],
        inputs={"pixel_size_um": PIXEL_SIZE_UM},
        position=PositionSpec(deepslice=True, bayesian=True),
        transform=TransformSpec(angles=True),
        nonlinear=NonlinearSpec(provider="openai-oauth"),
        agent_preprocessing=True,
    )


def tool_map(box: Any) -> dict[str, Any]:
    return {tool.__name__: tool for tool in box.tools}


def record_full_toolbox(rec: Recorder, folder: Path) -> tuple[list[str], Any]:
    from langslice.linear.engine import build_context
    from langslice.linear.job import ingest
    from langslice.linear.toolbox import build_tools

    spec = full_spec(folder)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    state = ingest(spec, ctx)
    box = build_tools(state, ctx, spec, image_model=stub_image_model(spec))
    t = tool_map(box)
    door = "tools"

    def call(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return rec.tool(door, t, name, *args, **kwargs)

    # Looking before anything is placed.
    call("status")
    call("view_slices", [ID0, ID1, ID2])
    call("view_slices", [ID1], view={"mode": "channels"})
    call("view_slices", [ID1], view={"channels": ["red", "green"], "zoom": [0.1, 0.1, 0.7, 0.8]})
    # A picture key given at the top level is refused by the strict check.
    call("view_slices", slices=[ID0], mode="section")
    call("view_atlas", [0.05, 0.2])
    call("view_atlas", [0.1], view={"atlas_channels": ["ara", "borders"],
                                    "regions": ["STR", "TH:left"],
                                    "border_color": "#00ffff", "border_thickness": 2.0})
    call("view_atlas", [0.1], view={"atlas_channels": ["borders"], "outlines": "outer"})
    call("view_atlas", [0.1], view={"atlas_channels": ["nissl"]})
    call("note", "Golden run: three synthetic sections.")

    # Positions, clamping, undo/redo.
    call("set_positions", [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
                           {"id": ID2, "position_mm": 0.2}])
    call("set_positions", [{"id": ID1, "position_mm": 0.15}], view={"mode": "overlay"})
    call("set_positions", [{"id": ID0, "position_mm": 9.0}])
    call("undo")
    call("redo")
    call("undo")
    call("undo")
    call("redo")

    # The placement viewer in every mode.
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}])
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}], view={"mode": "stacked"})
    call("view_placement", [{"id": ID0, "positions_mm": [0.05, 0.1]},
                            {"id": ID1, "positions_mm": [0.15]}],
         view={"mode": "side_by_side"})
    call("view_placement", [{"id": ID1, "positions_mm": [0.15]}],
         view={"mode": "overlay", "atlas_channels": ["ara", "borders"], "atlas_opacity": 0.3,
               "regions": ["CTX:right"],
               "zoom": [0.2, 0.2, 0.8, 0.9]})
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}], view={"mode": "checkerboard"})
    call("view_placement", [{"id": ID2, "positions_mm": [0.2]}],
         view={"mode": "outlines", "atlas_channels": ["ara", "borders"]})
    call("view_placement", [{"id": ID1, "positions_mm": [0.15]}],
         view={"mode": "section", "channels": ["green"]})
    call("view_stack")

    # Order, damage, orientation.
    call("reorder_slices", [ID2], after="start")
    call("reorder_slices", [ID0, ID1, ID2])
    call("mark_damaged", [{"id": ID2, "damaged": True, "note": "right third missing"}])
    call("mark_damaged", [{"id": ID1, "note": "test flag"}])
    call("mark_damaged", [{"id": ID1, "damaged": False}])
    call("orient_slices", [{"id": ID1, "flip": True, "rotate_deg": 90}])
    call("undo")

    # Appearance.
    call("preprocess", target="view", slices=[ID1], channel_weights=[1.0, 0.5, 0.0],
         clahe_clip=2.0)
    call("preprocess", target="fit", clahe_clip=6.0, clahe_tiles=4)
    call("preprocess", target="view", slices=[ID1], reset=True)

    # Cutting angles (then back), the position searches.
    call("set_cutting_angles", 1.0, 0.5)
    call("view_atlas", [0.1])
    call("undo")
    call("search_position", ID0, 0.1, False)
    call("run_deepslice", [ID0], False, [])

    # The in-plane transform: fits and direct adjustments.
    call("fit_affine", [])
    call("fit_affine", [ID0], method="silhouette", view={"mode": "outlines"})
    call("fit_affine", [ID1], exclude=["HY"], view={"mode": "side_by_side"})
    # Regions wholly inside the outline cannot steer a silhouette: refused.
    call("fit_affine", [ID0], method="silhouette", include=["TH"])
    call("fit_affine", [ID0], method="silhouette", exclude=["CTX:right"],
         view={"mode": "checkerboard"})
    call("adjust_transforms", [{"id": ID2, "rotation_deg": 3.0, "scale_x": 1.05, "scale_y": 0.98,
                                "translate_x_mm": 0.05, "translate_y_mm": -0.02,
                                "note": "surviving left half"}])
    call("adjust_transforms", [{"id": ID0, "rotation_deg": 1.0, "scale_x": 1.0, "scale_y": 1.01,
                                "translate_x_mm": 0.0, "translate_y_mm": 0.01}],
         view={"mode": "ab"})
    call("adjust_transforms", [{"id": ID1, "rotation_deg": -1.0, "scale_x": 0.99, "scale_y": 1.0,
                                "translate_x_mm": 0.02, "translate_y_mm": 0.0},
                               {"id": ID0, "rotation_deg": 1.0, "scale_x": 1.0, "scale_y": 1.01,
                                "translate_x_mm": 0.0, "translate_y_mm": 0.01}],
         view={"mode": "checkerboard", "zoom": [0.0, 0.0, 0.6, 0.6]})
    call("adjust_transforms", [{"id": ID2, "rotation_deg": 3.0, "scale_x": 1.05, "scale_y": 0.98,
                                "translate_x_mm": 0.05, "translate_y_mm": -0.02,
                                "pivot": "tissue"}], view={"mode": "side_by_side"})
    call("adjust_transforms", [{"id": ID1, "rotation_deg": -1.0, "scale_x": 0.99, "scale_y": 1.0,
                                "translate_x_mm": 0.02, "translate_y_mm": 0.0}],
         view={"mode": "outlines", "atlas_channels": ["ara", "borders"],
               "border_color": "magenta", "border_thickness": 0.5})
    call("adjust_transforms", [{"id": ID1, "rotation_deg": -1.0, "scale_x": 0.99, "scale_y": 1.0,
                                "translate_x_mm": 0.02, "translate_y_mm": 0.0}],
         view={"mode": "template"})
    call("undo")
    call("redo")
    call("submit", "Not yet.", [], [])

    # Nonlinear: traces (image model stubbed), the atlas lookup, deformable fits.
    call("trace_borders", ID0)
    call("grep_atlas", "TH")
    call("grep_atlas", "ST", section=ID0)
    call("fit_deformable", [ID0], fit_section="traced_borders", engine="ants")
    call("fit_deformable", [ID0], candidates=[
        {"fit_section": "traced_lines", "engine": "elastix"},
        {"stiffness": "soft", "engine": "elastix"},
    ])
    call("fit_deformable", [ID0])
    call("fit_deformable", [ID1], engine="elastix", include=["STR", "TH"], exclude=["HY"],
         view={"mode": "ab"})
    call("fit_deformable", [ID1], start="current", engine="elastix", stiffness="firm",
         view={"atlas_channels": ["ara", "borders"], "atlas_opacity": 0.4})
    call("fit_deformable", [ID2], keep_linear="Too little tissue survives for a warp.")
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}], view={"mode": "overlay"})
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}],
         view={"mode": "overlay", "deformation": "none"})
    call("status")
    call("submit", "Not yet: traces missing.", [], [])
    call("trace_borders", ID1)
    call("trace_borders", ID2, prompt="Trace only the surviving left half.")
    call("submit", "Placed, aligned and deformed three sections.", ["golden"], [])

    rec.raw("state", "final_main", {"state": state.to_dict()})
    return box.names, state


# --- door 1b: the toolbox at image resolution "auto", engine fixed, no image model ----


def record_auto_toolbox(rec: Recorder, folder: Path) -> list[str]:
    from langslice.linear.engine import build_context
    from langslice.linear.job import ingest
    from langslice.linear.spec import JobSpec, NonlinearSpec
    from langslice.linear.toolbox import build_tools

    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["position", "transform", "nonlinear"], inputs={"pixel_size_um": PIXEL_SIZE_UM},
        image_resolution="auto", nonlinear=NonlinearSpec(engine="elastix", provider="none"),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    state = ingest(spec, ctx)
    box = build_tools(state, ctx, spec)
    t = tool_map(box)
    door = "auto"

    def call(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return rec.tool(door, t, name, *args, **kwargs)

    call("view_slices", [ID0], view={"resolution": 200})
    call("set_positions", [{"id": ID0, "position_mm": 0.1}], view={"resolution": 160})
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}],
         view={"mode": "overlay", "resolution": 4000})
    call("view_stack", view={"resolution": 128})
    call("adjust_transforms", [{"id": ID0, "rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
                                "translate_x_mm": 0.0, "translate_y_mm": 0.0}],
         view={"resolution": 256})
    call("fit_deformable", [ID0], fit_section="traced_lines")
    call("fit_deformable", [ID0], view={"resolution": 300})
    return box.names


# --- door 1c: the look-before-commit gates and the delivery bookkeeping ----------------


class ToolContext:
    """The two things a tool reads from ADK's tool context: its call id, and
    the actions ``submit`` escalates."""

    def __init__(self, call_id: str) -> None:
        self.function_call_id = call_id
        self.actions = type("Actions", (), {"escalate": False})()

    def __repr__(self) -> str:
        return f"ToolContext({self.function_call_id!r})"


def record_gated_toolbox(rec: Recorder, folder: Path) -> list[str]:
    """Positions behind the gates, with pictures promoted to seen per model round.

    ``mark_placement_views_delivered`` is what the ADK driver calls once a
    model request carried a call's pictures; ``begin_model_call`` promotes
    direct calls (no tool context), as the MCP door does before every call.
    """
    from langslice.linear.engine import build_context
    from langslice.linear.job import ingest
    from langslice.linear.spec import JobSpec, PositionSpec
    from langslice.linear.toolbox import build_tools

    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["reorder", "position"], inputs={"pixel_size_um": PIXEL_SIZE_UM},
        position=PositionSpec(gated=True),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    state = ingest(spec, ctx)
    box = build_tools(state, ctx, spec)
    t = tool_map(box)
    door = "gated"

    def call(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return rec.tool(door, t, name, *args, **kwargs)

    def delivered(*ids: str) -> None:
        box.mark_placement_views_delivered(set(ids))
        rec.raw(door, "delivered", {"delivery_ids": sorted(ids)})

    # Not viewed yet: refused.
    call("set_positions", [{"id": ID0, "position_mm": 0.1}], tool_context=ToolContext("w0"))
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}], tool_context=ToolContext("c1"))
    # Same model round as the view: the write still returns its picture.
    call("set_positions", [{"id": ID0, "position_mm": 0.1}], tool_context=ToolContext("w1"))
    # A stale id from replayed history promotes nothing.
    delivered("old-call")
    delivered("c1", "w1")
    call("view_placement", [{"id": ID0, "positions_mm": [0.1]}, {"id": ID1, "positions_mm": [0.15]},
                            {"id": ID2, "positions_mm": [0.2]}], tool_context=ToolContext("c2"))
    # ID0 at 0.1 was delivered: suppressed. ID1's view is still pending: pictured.
    call("set_positions", [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15}],
         tool_context=ToolContext("w2"))
    # A direct call (no tool context), promoted at the next model-call boundary.
    call("view_placement", [{"id": ID2, "positions_mm": [0.2]}])
    box.begin_model_call()
    rec.raw(door, "begin_model_call", {})
    call("set_positions", [{"id": ID2, "position_mm": 0.2}])
    call("submit", "Not reviewed yet.", [], [], tool_context=ToolContext("s0"))
    call("view_stack")
    # A write needs a new view of that section, and a new review.
    call("set_positions", [{"id": ID1, "position_mm": 0.16}], tool_context=ToolContext("w3"))
    call("view_placement", [{"id": ID1, "positions_mm": [0.16]}], tool_context=ToolContext("c3"))
    call("set_positions", [{"id": ID1, "position_mm": 0.16}], tool_context=ToolContext("w4"))
    call("submit", "Still not reviewed.", [], [], tool_context=ToolContext("s1"))
    call("view_stack")
    call("submit", "Placed behind the gates.", [], [], tool_context=ToolContext("s2"))
    rec.raw("state", "final_gated", {"state": state.to_dict()})
    return box.names


# --- door 2: the engine's first model request ---------------------------------------


class _Captured(Exception):
    """Raised by the fake model once the first request is recorded."""


def record_engine_request(rec: Recorder, folder: Path) -> None:
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.models.registry import LLMRegistry

    from langslice.linear.engine import run

    captured: list[LlmRequest] = []

    class _FirstRequest(BaseLlm):
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            captured.append(llm_request)
            raise _Captured()
            yield  # pragma: no cover - makes this an async generator

    held = LLMRegistry.new_llm
    LLMRegistry.new_llm = staticmethod(lambda model: _FirstRequest(model=model))  # type: ignore[method-assign]
    try:
        asyncio.run(run(full_spec(folder), emit=lambda _m: None, atlas_loader=atlas_loader()))
    except _Captured:
        pass
    except Exception as exc:  # ADK may wrap the model's exception
        if not captured:
            raise RuntimeError("golden recorder: the engine never called the model") from exc
    finally:
        LLMRegistry.new_llm = held  # type: ignore[method-assign]
    if not captured:
        raise RuntimeError("golden recorder: the engine never called the model")
    request = captured[0]
    config = request.config
    system = config.system_instruction if config is not None else None
    if not isinstance(system, str):
        system = json.dumps(system.model_dump(exclude_none=True) if system is not None else None,
                            default=str)
    declarations = []
    for tool in (config.tools if config is not None else None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            declarations.append(declaration.model_dump(exclude_none=True, mode="json"))
    contents: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for content in request.contents or []:
        for part in content.parts or []:
            if part.inline_data is not None:
                contents.append({"role": content.role, "image": len(images),
                                 "mime": part.inline_data.mime_type})
                rec.sent.setdefault("engine", []).append(part.inline_data.data or b"")
                images.append(decode_image(part.inline_data.data or b""))
            elif part.text is not None:
                contents.append({"role": content.role, "text": part.text})
            else:
                contents.append({"role": content.role,
                                 "other": part.model_dump(exclude_none=True, mode="json")})
    rec.raw("engine", "job_statement", {"system_instruction": system})
    rec.raw("engine", "seed_message", {"contents": contents}, images)
    rec.raw("engine", "tool_declarations", {"tools": declarations})


# --- door 3: MCP ------------------------------------------------------------------------


def record_mcp(rec: Recorder, folder: Path) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from langslice.linear.spec import JobSpec
    from langslice.mcp_server.server import build_server

    positions = {ID0: 0.1, ID1: 0.15, ID2: 0.2}

    def spec_for(image_folder: str) -> JobSpec:
        # Positions from the host: the opening strips carry the atlas under
        # each section (the engine door covers the no-position opening).
        return JobSpec(image_folder=image_folder, preprocess="none", tasks=["transform"],
                       inputs={"pixel_size_um": PIXEL_SIZE_UM, "positions": positions})

    server = build_server(spec_for, str(folder), atlas_loader=atlas_loader())
    calls: list[tuple[str, dict[str, Any]]] = [
        ("status", {}),
        ("view_placement", {"entries": [{"id": ID0, "positions_mm": [0.1]}],
                            "view": {"mode": "overlay"}}),
        # Claude Desktop may send a nested object as a JSON string.
        ("view_slices", {"slices": [ID1], "view": '{"mode": "channels"}'}),
        ("view_slices", {"slices": [ID0], "mode": "section"}),
        ("fit_affine", {"slices": [ID1], "method": "silhouette"}),
        ("adjust_transforms", {"entries": [{"id": ID0, "rotation_deg": 2.0, "scale_x": 1.0,
                                            "scale_y": 1.0, "translate_x_mm": 0.0,
                                            "translate_y_mm": 0.0}]}),
        ("submit", {"summary": "done", "notes": [], "interval_breaks": []}),
    ]

    async def body(client: Any) -> list[tuple[str, dict[str, Any], list[Any]]]:
        out: list[tuple[str, dict[str, Any], list[Any]]] = []
        tools = sorted(tool.name for tool in (await client.list_tools()).tools)
        out.append(("list_tools", {}, [TextContent(type="text", text=json.dumps(tools))]))
        start = await client.call_tool("start_job", {})
        out.append(("start_job", {}, list(start.content)))
        first = start.content[0]
        count = int(re.search(r"show_stack\(page=(\d+)\)", first.text).group(1))
        for page in range(1, count + 1):
            shown = await client.call_tool("show_stack", {"page": page})
            out.append(("show_stack", {"page": page}, list(shown.content)))
        for name, arguments in calls:
            result = await client.call_tool(name, arguments)
            out.append((name, arguments, list(result.content)))
        return out

    async def main() -> list[tuple[str, dict[str, Any], list[Any]]]:
        async with create_connected_server_and_client_session(server) as client:
            return await body(client)

    for name, arguments, blocks in asyncio.run(main()):
        rec.blocks("mcp", name, {"tool": name, "arguments": arguments}, blocks)


# --- door 3b: checkpoint -> reload -> continue, through the MCP door ----------------------


def record_mcp_resume(rec: Recorder, folder: Path) -> None:
    """Two servers on one folder: the second resumes the first's checkpoint."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from langslice.linear.checkpoint import default_checkpoint_path, load_checkpoint
    from langslice.linear.spec import JobSpec
    from langslice.mcp_server.server import build_server

    positions = {ID0: 0.1, ID1: 0.15, ID2: 0.2}

    def spec_for(image_folder: str) -> JobSpec:
        return JobSpec(image_folder=image_folder, preprocess="none", tasks=["transform"],
                       resume=True,
                       inputs={"pixel_size_um": PIXEL_SIZE_UM, "positions": positions})

    first: list[tuple[str, dict[str, Any]]] = [
        ("adjust_transforms", {"entries": [{"id": ID0, "rotation_deg": 2.0, "scale_x": 1.0,
                                            "scale_y": 1.0, "translate_x_mm": 0.01,
                                            "translate_y_mm": 0.0}]}),
        ("fit_affine", {"slices": [ID1], "method": "silhouette"}),
        ("note", {"text": "First server: two sections aligned."}),
    ]
    second: list[tuple[str, dict[str, Any]]] = [
        ("start_job", {}),
        ("status", {}),
        ("undo", {}),
        ("redo", {}),
        ("adjust_transforms", {"entries": [{"id": ID2, "rotation_deg": -1.0, "scale_x": 1.0,
                                            "scale_y": 1.0, "translate_x_mm": 0.0,
                                            "translate_y_mm": 0.0}]}),
        ("submit", {"summary": "resumed and finished", "notes": [], "interval_breaks": []}),
    ]

    def session(label: str, calls: list[tuple[str, dict[str, Any]]]) -> None:
        server = build_server(spec_for, str(folder), atlas_loader=atlas_loader())

        async def main() -> list[tuple[str, dict[str, Any], list[Any]]]:
            out: list[tuple[str, dict[str, Any], list[Any]]] = []
            async with create_connected_server_and_client_session(server) as client:
                for name, arguments in calls:
                    result = await client.call_tool(name, arguments)
                    out.append((name, arguments, list(result.content)))
            return out

        for name, arguments, blocks in asyncio.run(main()):
            rec.blocks("resume", f"{label}_{name}", {"tool": name, "arguments": arguments},
                       blocks)

    session("first", first)
    session("second", second)
    saved = load_checkpoint(default_checkpoint_path(str(folder)))
    rec.raw("state", "final_resumed", {"state": saved.to_dict() if saved else None})


# --- every door's declarations ----------------------------------------------------------


def record_declarations(rec: Recorder, folder: Path) -> None:
    """What each door declares, per tool: the ADK function declarations of
    three toolboxes (the full spec, the "auto" spec with the engine fixed and
    no image model, and a positioning-only gated spec) and the MCP tool list
    (name, description, input schema, annotations) of a transform job and a
    positioning job. Recorded after everything else, so no earlier entry is
    renumbered."""
    import dataclasses

    from google.adk.tools import FunctionTool
    from mcp.shared.memory import create_connected_server_and_client_session

    from langslice.adk.media import packaged_tools
    from langslice.linear.engine import build_context
    from langslice.linear.job import ingest
    from langslice.linear.spec import JobSpec, NonlinearSpec, PositionSpec
    from langslice.linear.toolbox import build_tools
    from langslice.mcp_server.server import build_server

    base = {"model": "fake-model", "preprocess": "none",
            "inputs": {"pixel_size_um": PIXEL_SIZE_UM}}
    specs = {
        "full": full_spec(folder),
        "auto": JobSpec(image_folder=str(folder), tasks=["position", "transform", "nonlinear"],
                        image_resolution="auto",
                        nonlinear=NonlinearSpec(engine="elastix", provider="none"), **base),
        "gated": JobSpec(image_folder=str(folder), tasks=["position"],
                         position=PositionSpec(gated=True), **base),
    }
    for label, spec in specs.items():
        ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
        box = build_tools(ingest(spec, ctx), ctx, spec,
                          image_model=stub_image_model(spec) if label == "full" else None)
        declarations = [FunctionTool(tool)._get_declaration().model_dump(  # noqa: SLF001
            exclude_none=True, mode="json") for tool in packaged_tools(box.tools)]
        rec.raw("declarations", f"adk_{label}", {"tools": declarations})

    for label, tasks in (("transform", ["transform"]), ("position", ["reorder", "position"])):
        spec = JobSpec(image_folder=str(folder), tasks=tasks, **base)
        server = build_server(lambda image_folder, spec=spec: dataclasses.replace(
            spec, image_folder=image_folder), str(folder), atlas_loader=atlas_loader())

        async def listed(server: Any = server) -> list[dict[str, Any]]:
            async with create_connected_server_and_client_session(server) as client:
                return [tool.model_dump(mode="json", exclude_none=True)
                        for tool in (await client.list_tools()).tools]

        rec.raw("declarations", f"mcp_{label}", {"tools": asyncio.run(listed())})


# --- the job folders ------------------------------------------------------------------


def _pixels_digest(array: np.ndarray) -> str:
    import hashlib

    data = np.ascontiguousarray(array)
    head = f"{data.dtype.str}{list(data.shape)}".encode()
    return hashlib.sha256(head + data.tobytes()).hexdigest()


def snapshot_job_folders(rec: Recorder, folders: dict[str, Path]) -> None:
    """Each door's job folder after the run (see the module text)."""
    import hashlib

    import tifffile

    from langslice.job.layout import JOB_DIRNAME, read_job_file
    from langslice.job.layout import JobLayout as Layout
    from langslice.job.views import flush_all

    flush_all()
    summary: dict[str, Any] = {}
    for door, folder in folders.items():
        root = folder / JOB_DIRNAME
        files = sorted(rec.norm.text(path.relative_to(root).as_posix())
                       for path in root.rglob("*") if path.is_file())
        index_path = root / "views.jsonl"
        entries = ([json.loads(line) for line in index_path.read_text().splitlines() if line]
                   if index_path.exists() else [])
        layers: dict[str, Any] = {}
        for entry in entries:
            view = root / entry["path"]
            record = json.loads((view / "view.json").read_text())
            digest = {"view_json": hashlib.sha256(
                canonical(rec.norm.data(record)).encode()).hexdigest()}
            if (view / "labels.tif").exists():
                labels = tifffile.imread(view / "labels.tif")
                with Image.open(view / "borders.png") as opened:
                    borders = np.asarray(opened)
                digest.update(labels=_pixels_digest(labels), borders=_pixels_digest(borders),
                              labels_dtype=str(labels.dtype), size=list(labels.shape[::-1]))
            layers[rec.norm.text(entry["path"])] = digest
        saved = [(root / entry["path"] / "view.jpg").read_bytes() for entry in entries]
        sent = rec.sent.get(door, [])
        job_file = read_job_file(Layout(root)) or {}
        job_file.pop("created_at", None)
        # The public files derived from the state (formats phase):
        # registration.json on every write, the maps and exports at submit.
        derived: dict[str, Any] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if "/views/" in relative or "/deformable/" in relative or not path.is_file():
                continue
            name = path.name
            if name in ("registration.json", "maps.json", "quicknii.json", "visualign.json"):
                data = rec.norm.data(json.loads(path.read_text()))
                derived[rec.norm.text(relative)] = hashlib.sha256(
                    canonical(data).encode()).hexdigest()
            elif name in ("coords.tif", "labels.tif", "labels_fiji.tif", "residual.tif"):
                pixels = tifffile.imread(path)
                derived[rec.norm.text(relative)] = {
                    "pixels": _pixels_digest(pixels), "dtype": str(pixels.dtype),
                    "shape": list(pixels.shape)}
            elif name == "labels.csv":
                derived[rec.norm.text(relative)] = path.read_text()
        summary[door] = {
            "derived": derived,
            "files": files,
            "views_index": rec.norm.data(entries),
            "job_file": rec.norm.data(job_file),
            "layers": layers,
            "pictures_sent": len(sent),
            "saved_jpegs_equal_sent": saved == sent,
        }
    rec.raw("job", "folders", {"folders": summary})


# --- everything -------------------------------------------------------------------------


def record(out: Path) -> dict[str, Any]:
    """Run every door into *out* (emptied first); return the run summary."""
    for variable in ("LANGSLICE_TRACE_DIR", "LANGSLICE_VLM_DEBUG_DIR"):
        os.environ.pop(variable, None)
    apply_patches()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="langslice-golden-") as temporary:
        root = Path(temporary)
        rec = Recorder(out, root)
        folders = {name: root / name
                   for name in ("tools", "auto", "gated", "engine", "mcp", "resume")}
        for folder in folders.values():
            write_sections(folder)
        main_names, _state = record_full_toolbox(rec, folders["tools"])
        auto_names = record_auto_toolbox(rec, folders["auto"])
        gated_names = record_gated_toolbox(rec, folders["gated"])
        record_engine_request(rec, folders["engine"])
        record_mcp(rec, folders["mcp"])
        record_mcp_resume(rec, folders["resume"])
        snapshot_job_folders(rec, folders)
        declared = root / "declarations"
        write_sections(declared)
        record_declarations(rec, declared)
        built = sorted(set(main_names) | set(auto_names) | set(gated_names))
        missing = sorted(set(built) - rec.called)
        if missing:
            raise RuntimeError(
                f"golden recorder: tools built but never called: {missing}; add them to the "
                "scenario in tests/golden/record.py and regenerate")
        summary = {
            "tools_built": built,
            "tools_full_spec": sorted(main_names),
            "tools_auto_spec": sorted(auto_names),
            "tools_gated_spec": sorted(gated_names),
            "calls": rec.count,
        }
        (out / "manifest.json").write_text(canonical(summary), encoding="utf-8")
        print(f"recorded {rec.count} entries in {time.perf_counter() - rec.started:.1f} s "
              f"-> {out}", file=sys.stderr)
        return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR,
                        help="where to write (default: the checked-in goldens)")
    record(parser.parse_args(argv).out)


if __name__ == "__main__":
    main()
