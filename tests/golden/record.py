"""Golden snapshots of every linear tool: the recorder.

A fixed, scripted sequence of tool calls on a small synthetic stack, recorded
so a change can show that the pictures stay pixel-identical and the data and
text stay equal.

What is recorded, per call: every picture (decoded and saved as PNG), the
structured data the tool returned (JSON), and the text the model receives.
The doors covered:

- ``tools``: the tools ``build_tools`` returns, called directly, on a job
  opened as the engine opens it. The full spec (every task, every optional
  tool, the image model stubbed, picture size ``low``): every tool, every
  ``look`` mode, ``zoom`` (of a zoom too), the channel tools, damage,
  positions with undo/redo, the in-plane transform by hand and by elastix,
  the background ``trace_borders`` and its notice, ``ants_syn``, the
  job-folder files, refusals, ``submit`` with ``left_linear``. Then an
  ``auto`` picture-size spec with no image model.
- ``gated``: the look-before-commit gates on a ``position.gated`` toolbox:
  a position write refused until the section was looked at in mode overlay
  or positioning, submit refused until a positioning look of the stack.
- ``long``: eight sections, so the positioning picture is split into parts
  and ``look`` names the pictures past four under ``not_shown``.
- ``engine``: ``langslice.agent.engine.run`` with a fake ADK model that
  records the first model request (the job statement, the opening strips and
  the status table, the tool declarations) and stops the session.
- ``mcp``: the MCP server driven by an in-memory client (``start_job``,
  ``show_stack`` pages and a few tools, a retired tool's name), as Claude
  hosts see it.
- ``resume``: a checkpoint -> reload -> continue round trip through the MCP
  door: one server writes, a second server on the same folder resumes from
  the checkpoint and carries on (including ``undo``/``redo`` across it).
- the final stack state of the main toolbox, the gated one and the resumed job.
- ``declarations``: what each door declares per tool (ADK function
  declarations of four toolboxes, the last a host forcing the change tools'
  pictures and requiring a deformation; MCP tool lists with their schemas).
- ``mcp_*``: three more MCP jobs, recorded last (nonlinear with the image
  model connected and not, image resolution "auto" with the opening-read
  gate and a clamped picture size).
- ``job_folders``: each door's job folder (``<images>/langslice``) after
  the run: its file list, the views index, ``job.json``, a hash of every
  saved picture's label and border layers (decoded pixels) and of its
  ``view.json``, whether every saved ``view.jpg`` is byte for byte the
  JPEG the door sent, in order, and the derived public files
  (``registration.json``, each section's maps, the exports: decoded pixels
  and normalised JSON hashed, ``labels.csv`` as text).

Everything goes through public entry points (``build_tools``, ``engine.run``,
``doors.mcp.server.build_server``). The few internals touched are listed in
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
    ("langslice.core.deformation", "USE_PROCESS_POOL", False),
    ("langslice.core.deformation", "DETAIL_LEVEL", "coarse"),
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

    from langslice.core.nonlinear.types import GeneratedSegmentation
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
        message parts (:func:`langslice.doors.tools.media.package_result`), which
        are what is recorded.
        """
        from google.genai import types

        from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
        from langslice.doors.tools.media import package_result

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
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec

    return JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["reorder", "position", "transform", "nonlinear"],
        inputs={"pixel_size_um": PIXEL_SIZE_UM},
        position=PositionSpec(),
        transform=TransformSpec(angles=True),
        nonlinear=NonlinearSpec(provider="openai-oauth"),
        agent_preprocessing=True,
    )


def tool_map(box: Any) -> dict[str, Any]:
    return {tool.__name__: tool for tool in box.tools}


def open_toolbox(spec: Any, image_model: Any = None) -> Any:
    """The toolbox of *spec*'s job, opened as the engine opens it
    (``Job.open``: the job folder, the starting positions)."""
    from langslice.agent.engine import build_context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import Job

    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=atlas_loader())
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    return build_tools(job.state, ctx, spec, job=job, image_model=image_model)


def record_full_toolbox(rec: Recorder, folder: Path) -> tuple[list[str], Any]:
    """Every tool of the full spec, in the order a run might call them: looking
    (every ``look`` mode, ``zoom`` and a zoom of a zoom, the channel tools, the
    atlas lookups), positions and cutting angles with undo/redo, damage, the
    in-plane transform by hand and by elastix, the background trace and its
    notice, ANTs SyN on top of the current registration, the job-folder files,
    refusals along the way, and ``submit`` with ``left_linear``."""
    spec = full_spec(folder)
    box = open_toolbox(spec, stub_image_model(spec))
    state = box.job.state
    t = tool_map(box)
    door = "tools"

    def call(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return rec.tool(door, t, name, *args, **kwargs)

    def landed() -> None:
        """Wait for the background work, so its notice opens the next reply."""
        box.job.background.wait_all()

    # Looking at the starting positions (evenly spaced, not yet placed).
    call("status")
    call("look", "section")
    call("look", "section", sections=[ID1], channels=["red", "green"])
    # Zoom: the newest picture that is not a zoom, then a zoom of that zoom,
    # then the first picture by its number, then a number that is not there.
    zoomed = call("zoom", [40, 20, 200, 140])
    call("zoom", [120, 80, 520, 380], picture=int(zoomed["pictures"][0]["id"]))
    call("zoom", [30, 40, 150, 130], picture=1)
    call("zoom", [0, 0, 50, 50], picture=999)
    call("look", "atlas", positions_mm=[0.05, 0.2])
    call("look", "atlas", positions_mm=[0.1], atlas_layers=["template", "borders"])
    call("look", "atlas", positions_mm=[0.1], atlas_layers=["ara"])  # "ara" read as template
    call("look", "positioning")
    # Refusals: sections named by a number and by a name that is no file's
    # (sections are named by filename), a stray top-level argument (the
    # strict check), a channel the section does not have.
    call("look", "section", sections=[0, "nope.png"])
    call("look", "section", sections=[ID0], view={"mode": "section"})
    call("look", "section", sections=[ID0], channels=["purple"])
    # Display settings: display only, they persist and every caption states them.
    call("set_channel_properties", "red", contrast_limits=[20.0, 200.0], gamma=1.4,
         colormap="magenta")
    call("look", "section", sections=[ID1])
    call("set_channel_properties", "purple")
    # The preprocessed channel: before / after pictures, its recipe, a look at it.
    call("set_preprocessed_channel_properties", sections=[ID1],
         channel_weights=[1.0, 0.5, 0.0], clahe_clip=2.0)
    call("set_preprocessed_channel_properties", clahe_clip=6.0, clahe_tiles=4)
    call("look", "section", sections=[ID1], channels=["preprocessed"])
    call("set_preprocessed_channel_properties", sections=[ID1], reset=True)
    # The atlas, by text and by picture.
    call("grep_atlas", "TH")
    call("grep_atlas_view", ["STR", "TH:left"], [0.1, 0.4])
    call("grep_atlas_view", ["NOPE"], [0.1])
    call("note", "Golden run: three synthetic sections.")

    # Positions (the order follows them), clamping, cutting angles, undo/redo.
    call("position_sections", [{"id": ID0, "position_mm": 0.1},
                               {"id": ID1, "position_mm": 0.15},
                               {"id": ID2, "position_mm": 0.2}])
    call("position_sections", [{"id": ID0, "position_mm": 9.0}])
    call("undo")
    call("redo")
    call("undo")
    call("position_sections", [{"id": ID2, "position_mm": 0.05}])  # reorders the stack
    call("undo")
    call("position_sections", cutting_angles={"pitch_deg": 1.0, "yaw_deg": 0.5}, view=False)
    call("look", "atlas", positions_mm=[0.1])
    call("undo")
    call("position_sections", [{"id": ID0, "position_mm": 0.1, "mm": 1}])  # a stray key

    # Every look mode on the placed stack.
    call("look", "overlay", sections=[ID0])
    call("look", "overlay", sections=[ID1], atlas_layers=["template", "borders"],
         atlas_opacity=0.3)
    call("look", "positioning", positions_mm=[0.1, 0.2])
    call("look", "positioning", sections=[ID0, ID2])

    # Damage by atlas region: shaded on the section and on the atlas.
    call("mark_damage", ID2, ["CTX:right"], note="right third missing")
    call("mark_damage", ID1, ["NOPE"])
    call("mark_damage", ID1, ["HY"], note="test mark")
    call("mark_damage", "s1", [])  # the filename without its extension

    # The in-plane transform by hand: absolute values, a value left out kept.
    call("interactive_transform", [{"id": ID1, "flip": True, "rotate_quarter": 90}])
    call("undo")
    call("interactive_transform", [{"id": ID2, "rotation_deg": 3.0, "scale_x": 1.05,
                                    "scale_y": 0.98, "translate_x_mm": 0.05,
                                    "translate_y_mm": -0.02}])
    call("interactive_transform", [{"id": ID2, "translate_x_mm": 0.0}])  # the rest kept
    call("interactive_transform", [{"id": ID0, "rotation_deg": 1.0},
                                   {"id": ID1, "shear": 0.02}], view=False)
    call("interactive_transform", [{"id": ID0, "rotate_quarter": 45}])
    # elastix: every placed section, then one by a region, with "ara" for template.
    call("elastix_affine")
    call("elastix_affine", [ID1], restrict_to=["TH"], atlas_image="ara")
    call("elastix_affine", [ID0], restrict_to=["NOPE"])
    call("undo")
    call("redo")
    # Nonlinear on: a section must be deformed or named in left_linear.
    call("submit", "Not yet.", [], [])

    # The packaged trace runs in the background; its notice opens the next reply.
    call("trace_borders", ID0)
    landed()
    call("status")
    call("trace_borders", ID0)  # already fitted at this placement: nothing again
    # ANTs SyN on top of the current registration; damage left out by itself.
    call("ants_syn", [ID1])
    call("ants_syn", [ID1], restrict_to=["TH"], stiffness="soft", atlas_image="ara")
    call("ants_syn", [ID2], view=False)
    call("undo")  # ID2 is left linear (submit's left_linear)
    call("ants_syn", [ID0, ID1, ID2, 0, 1])
    call("look", "overlay", sections=[ID0, ID1])
    call("look", "overlay", sections=[ID0], warp="none")
    call("trace_borders", ID1, prompt="Trace only the thalamus.", restrict_to=["TH"])
    landed()
    call("note", "Second trace landed.")

    # The job folder: list, search, read, a picture's index entry, outside refused.
    call("list_files")
    call("list_files", "views", pattern="*.json")
    call("search_files", "Golden run")
    call("read_file", "views.jsonl", limit=3)
    call("read_file", "sections/s0/views/000001_look_section/view.jpg")
    call("read_file", "../outside.txt")
    call("status")
    call("submit", "Placed, aligned and deformed three sections.", ["golden"], [],
         left_linear=[{"id": ID2, "reason": "Too little tissue survives for a warp."}])

    rec.raw("state", "final_main", {"state": state.to_dict()})
    return box.names, state


# --- door 1b: the toolbox at image resolution "auto", no image model -------------------


def record_auto_toolbox(rec: Recorder, folder: Path) -> list[str]:
    from langslice.core.spec import JobSpec, NonlinearSpec

    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["position", "transform", "nonlinear"], inputs={"pixel_size_um": PIXEL_SIZE_UM},
        image_resolution="auto", nonlinear=NonlinearSpec(provider="none"),
    )
    box = open_toolbox(spec)
    t = tool_map(box)
    door = "auto"

    def call(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return rec.tool(door, t, name, *args, **kwargs)

    call("look", "section", sections=[ID0], resolution=200)
    call("look", "section", sections=[ID0], resolution=4000)  # clamped to the lane
    call("position_sections", [{"id": ID0, "position_mm": 0.1}])
    call("look", "overlay", sections=[ID0], resolution=300)
    call("interactive_transform", [{"id": ID0, "rotation_deg": 0.0}])
    call("ants_syn", [ID0])
    call("look", "overlay", sections=[ID0], resolution=256)
    return box.names


# --- door 1c: the look-before-commit gates ---------------------------------------------


def record_gated_toolbox(rec: Recorder, folder: Path) -> list[str]:
    """Positions behind the gates: a write refused until the section was looked
    at in mode overlay or positioning, submit refused until a positioning look
    of the whole stack since the last write."""
    from langslice.core.spec import JobSpec, PositionSpec

    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["reorder", "position"], inputs={"pixel_size_um": PIXEL_SIZE_UM},
        position=PositionSpec(gated=True),
    )
    box = open_toolbox(spec)
    state = box.job.state
    t = tool_map(box)
    door = "gated"

    def call(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return rec.tool(door, t, name, *args, **kwargs)

    every = [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
             {"id": ID2, "position_mm": 0.2}]
    # Not looked at yet: refused. One overlay look lets that section's write
    # through; a positioning look of the stack compares every section.
    call("position_sections", every)
    call("look", "overlay", sections=[ID0])
    call("position_sections", every, view=False)  # ID0 written, the others rejected
    call("look", "positioning")
    call("position_sections", every)  # every section compared by that look
    call("submit", "Not reviewed yet.", [], [], tool_context=ToolContext("s0"))
    call("look", "positioning")
    call("submit", "Placed behind the gates.", [], [], tool_context=ToolContext("s1"))
    rec.raw("state", "final_gated", {"state": state.to_dict()})
    return box.names


class ToolContext:
    """The two things a tool reads from ADK's tool context: its call id, and
    the actions ``submit`` escalates."""

    def __init__(self, call_id: str) -> None:
        self.function_call_id = call_id
        self.actions = type("Actions", (), {"escalate": False})()

    def __repr__(self) -> str:
        return f"ToolContext({self.function_call_id!r})"


# --- door 1d: a long stack, its positioning picture in parts ---------------------------

#: More sections than one positioning picture holds (``core.positioning.PER_PICTURE``).
LONG_STACK = tuple(f"long{index}.png" for index in range(8))


def record_long_stack(rec: Recorder, folder: Path) -> list[str]:
    """Eight sections: ``position_sections`` and ``look`` positioning split the
    stack into full-size parts, in position order."""
    import cv2

    from langslice.core.positioning import PER_PICTURE
    from langslice.core.spec import JobSpec
    from tests.deformable_synthetic import SyntheticAtlas, bump_field, render_section

    if len(LONG_STACK) <= PER_PICTURE:
        raise RuntimeError("golden recorder: the long stack no longer needs two parts")
    folder.mkdir(parents=True, exist_ok=True)
    atlas = SyntheticAtlas()
    for seed, name in enumerate(LONG_STACK):
        image, _ = render_section(atlas, bump_field([(170, 110, 30, 0.02 * (seed % 3 - 1),
                                                      0.0)]), seed=seed)
        plane = cv2.GaussianBlur(np.asarray(image)[..., 0], (0, 0), 1.2)
        plane = cv2.resize(plane, (136, 104), interpolation=cv2.INTER_AREA)
        Image.fromarray(np.stack([plane] * 3, axis=-1)).save(folder / name)
        os.utime(folder / name, (FIXED_MTIME, FIXED_MTIME))
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   tasks=["position"], inputs={"pixel_size_um": 2 * PIXEL_SIZE_UM})
    box = open_toolbox(spec)
    t = tool_map(box)
    # Positions out of file order: the picture and the stack follow position.
    positions = [0.2, 0.02, 0.05, 0.24, 0.08, 0.11, 0.17, 0.14]
    rec.tool("long", t, "position_sections", [
        {"id": name, "position_mm": value} for name, value in zip(LONG_STACK, positions,
                                                                  strict=True)])
    rec.tool("long", t, "look", "positioning", positions_mm=[0.05, 0.15])
    # More sections than one look shows: the rest named with the call that draws them.
    rec.tool("long", t, "look", "section")
    return box.names


# --- door 2: the engine's first model request ---------------------------------------


class _Captured(Exception):
    """Raised by the fake model once the first request is recorded."""


def record_engine_request(rec: Recorder, folder: Path) -> None:
    from google.adk.models import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.models.registry import LLMRegistry

    from langslice.agent.engine import run

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

    from langslice.core.spec import JobSpec
    from langslice.doors.mcp.server import build_server

    positions = {ID0: 0.1, ID1: 0.15, ID2: 0.2}

    def spec_for(image_folder: str) -> JobSpec:
        # Positions from the host: the opening strips carry the atlas under
        # each section (the engine door covers the no-position opening).
        return JobSpec(image_folder=image_folder, preprocess="none", tasks=["transform"],
                       inputs={"pixel_size_um": PIXEL_SIZE_UM, "positions": positions})

    server = build_server(spec_for, str(folder), atlas_loader=atlas_loader())
    calls: list[tuple[str, dict[str, Any]]] = [
        ("status", {}),
        ("look", {"mode": "overlay", "sections": [ID0]}),
        # A retired tool answers with its replacement; nothing is done.
        ("view_slices", {"slices": [ID1]}),
        # A stray argument is refused before the tool runs.
        ("look", {"mode": "section", "sections": [ID0], "view": {"mode": "section"}}),
        ("elastix_affine", {"sections": [ID1]}),
        # Claude Desktop may send a nested object as a JSON string.
        ("interactive_transform", {"sections": json.dumps([{"id": ID0, "rotation_deg": 2.0}])}),
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

    from langslice.core.spec import JobSpec
    from langslice.doors.mcp.server import build_server
    from langslice.job.checkpoint import load_checkpoint
    from langslice.job.layout import JobLayout

    positions = {ID0: 0.1, ID1: 0.15, ID2: 0.2}

    def spec_for(image_folder: str) -> JobSpec:
        return JobSpec(image_folder=image_folder, preprocess="none", tasks=["transform"],
                       resume=True,
                       inputs={"pixel_size_um": PIXEL_SIZE_UM, "positions": positions})

    first: list[tuple[str, dict[str, Any]]] = [
        ("interactive_transform", {"sections": [{"id": ID0, "rotation_deg": 2.0,
                                                 "translate_x_mm": 0.01}]}),
        ("elastix_affine", {"sections": [ID1]}),
        ("note", {"text": "First server: two sections aligned."}),
    ]
    second: list[tuple[str, dict[str, Any]]] = [
        ("start_job", {}),
        # A new conversation reads the opening before it writes (the gate).
        ("show_stack", {"page": 1}),
        ("status", {}),
        ("undo", {}),
        ("redo", {}),
        ("interactive_transform", {"sections": [{"id": ID2, "rotation_deg": -1.0}]}),
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
    saved = load_checkpoint(str(JobLayout.for_images(folder).state_file))
    rec.raw("state", "final_resumed", {"state": saved.to_dict() if saved else None})


# --- every door's declarations ----------------------------------------------------------


def record_declarations(rec: Recorder, folder: Path) -> None:
    """What each door declares, per tool: the ADK function declarations of
    four toolboxes (the full spec; the "auto" spec without an image model;
    a positioning-only gated spec; a host that forces the change tools'
    pictures and requires a deformation on every section) and the MCP tool
    list (name, description, input schema, annotations) of a transform job
    and a positioning job."""
    import dataclasses

    from google.adk.tools import FunctionTool
    from mcp.shared.memory import create_connected_server_and_client_session

    from langslice.agent.engine import build_context
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec
    from langslice.doors.mcp.server import build_server
    from langslice.doors.tools.media import packaged_tools
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest

    base = {"model": "fake-model", "preprocess": "none",
            "inputs": {"pixel_size_um": PIXEL_SIZE_UM}}
    specs = {
        "full": full_spec(folder),
        "auto": JobSpec(image_folder=str(folder), tasks=["position", "transform", "nonlinear"],
                        image_resolution="auto", nonlinear=NonlinearSpec(provider="none"),
                        **base),
        "gated": JobSpec(image_folder=str(folder), tasks=["position"],
                         position=PositionSpec(gated=True), **base),
        "host": JobSpec(image_folder=str(folder), tasks=["transform", "nonlinear"],
                        force_view=True, agent_damage=False,
                        nonlinear=NonlinearSpec(provider="none", require_deformation=True),
                        **base),
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
            spec, image_folder=image_folder), str(folder), atlas_loader=atlas_loader(),
            fresh=True)

        async def listed(server: Any = server) -> list[dict[str, Any]]:
            async with create_connected_server_and_client_session(server) as client:
                return [tool.model_dump(mode="json", exclude_none=True)
                        for tool in (await client.list_tools()).tools]

        rec.raw("declarations", f"mcp_{label}", {"tools": asyncio.run(listed())})


# --- MCP variants: nonlinear with and without the image model, auto picture size ------


def record_mcp_variants(rec: Recorder, root: Path) -> None:
    """Three more MCP jobs: the nonlinear task on a supplied linear placement
    with its image model connected (the stub's lane; no image call is made)
    and with it not connected (no trace_borders; the statement says the tool
    is off), and image resolution "auto" (the resolution range, a clamped
    request), each with its tool list and start_job statement; the last also
    shows the opening-read gate refusing a write before show_stack."""
    import dataclasses

    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from langslice.core.spec import JobSpec, NonlinearSpec
    from langslice.doors.api import setup
    from langslice.doors.mcp.server import build_server

    positions = {ID0: 0.1, ID1: 0.15, ID2: 0.2}
    supplied = {"pixel_size_um": PIXEL_SIZE_UM, "positions": positions, "transforms": {
        name: {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               "mirrored": False} for name in positions}}
    variants: list[tuple[str, JobSpec, bool, list[tuple[str, dict[str, Any]]]]] = [
        ("nonlinear_traced", JobSpec(
            image_folder="", preprocess="none", tasks=["nonlinear"], inputs=supplied,
            nonlinear=NonlinearSpec(provider="openai-oauth")), True, []),
        ("nonlinear_off", JobSpec(
            image_folder="", preprocess="none", tasks=["nonlinear"], inputs=supplied,
            nonlinear=NonlinearSpec(provider="openai-oauth")), False, []),
        ("auto", JobSpec(
            image_folder="", preprocess="none", tasks=["transform"], image_resolution="auto",
            inputs={"pixel_size_um": PIXEL_SIZE_UM, "positions": positions}), True, [
                ("note", {"text": "Before the opening."}),
                ("show_stack", {"page": 1}),
                ("note", {"text": "After the opening."}),
                ("look", {"mode": "section", "sections": [ID1], "resolution": 5000}),
            ]),
    ]
    held = setup.image_model_connected
    try:
        for label, spec, linked, calls in variants:
            folder = root / label
            write_sections(folder)
            setup.image_model_connected = lambda _provider, linked=linked: linked  # type: ignore[assignment]
            server = build_server(lambda image_folder, spec=spec: dataclasses.replace(
                spec, image_folder=image_folder), str(folder), atlas_loader=atlas_loader())

            async def body(server: Any = server, calls: Any = calls,
                           ) -> list[tuple[str, dict[str, Any], list[Any]]]:
                out: list[tuple[str, dict[str, Any], list[Any]]] = []
                async with create_connected_server_and_client_session(server) as client:
                    listed = (await client.list_tools()).tools
                    out.append(("list_tools", {}, [TextContent(type="text", text=json.dumps(
                        sorted(tool.name for tool in listed)))]))
                    for tool in listed:
                        if tool.name in ("trace_borders", "ants_syn", "look"):
                            out.append((f"declare_{tool.name}", {}, [TextContent(
                                type="text", text=json.dumps(tool.model_dump(
                                    mode="json", exclude_none=True), sort_keys=True))]))
                    start = await client.call_tool("start_job", {})
                    out.append(("start_job", {}, list(start.content)))
                    for name, arguments in calls:
                        result = await client.call_tool(name, arguments)
                        out.append((name, arguments, list(result.content)))
                return out

            for name, arguments, blocks in asyncio.run(body()):
                rec.blocks(f"mcp_{label}", name, {"tool": name, "arguments": arguments}, blocks)
    finally:
        setup.image_model_connected = held  # type: ignore[assignment]


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
    os.environ.pop("LANGSLICE_TRACE_DIR", None)
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
        long_names = record_long_stack(rec, root / "long")
        record_engine_request(rec, folders["engine"])
        record_mcp(rec, folders["mcp"])
        record_mcp_resume(rec, folders["resume"])
        snapshot_job_folders(rec, folders)
        declared = root / "declarations"
        write_sections(declared)
        record_declarations(rec, declared)
        record_mcp_variants(rec, root / "variants")
        region_folder = root / "region_zoom"
        write_sections(region_folder)
        region_spec = full_spec(region_folder)
        region_box = open_toolbox(region_spec, stub_image_model(region_spec))
        region_tools = tool_map(region_box)
        rec.tool("regions", region_tools, "look", "overlay", sections=[ID1])
        rec.tool("regions", region_tools, "zoom", region="STR")
        rec.tool("regions", region_tools, "zoom", region="TH:left")
        rec.tool("regions", region_tools, "zoom", box=[0, 0, 50, 50], region="STR")
        built = sorted(set(main_names) | set(auto_names) | set(gated_names)
                       | set(long_names))
        from langslice.ops.registry import VERBS

        unbuilt = sorted(name for name, verb in VERBS.items()
                         if not verb.scripting and name not in built)
        if unbuilt:
            raise RuntimeError(f"golden recorder: tools no toolbox here builds: {unbuilt}")
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
