"""The shared synthetic stack: three sections of the synthetic atlas, its
loader, the fit settings every test runs with, the stand-in image model, and
the normaliser the agent-text snapshot uses.

``tests/conftest.py`` applies :data:`PATCHES` once per test process (fits in
process at the coarse level, every atlas the adult mouse brain's size), so
every test sees the same settings whatever
ran before it; a child interpreter (``tests/cli_child.py``) calls
:func:`apply_patches` itself.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

#: Stand-in for the temporary folder every run works in.
ROOT_TOKEN = "<ROOT>"
#: Stand-in for a hex digest.
HEX_TOKEN = "<hex>"
#: Hex runs this long or longer are digests (cache keys and record folders
#: derive from absolute paths and mtimes, never stable across folders).
HEX_RUN = re.compile(r"[0-9a-f]{16,}")
#: Keys whose values are wall-clock durations, normalised to this token.
SECONDS_TOKEN = "<seconds>"
SECONDS_KEYS = frozenset({"runtime_s", "elapsed_s", "seconds", "wall_s"})

#: Fixed mtime for the section files (2026-01-01 UTC), so nothing derived
#: from it varies.
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
        # every picture is smaller.
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


#: Module attributes every test runs with: fits run in this process (no
#: spawn pool, whose start-up costs seconds per call) at the coarse detail
#: level. Same code path, smaller pyramid.
def _reference_brain(atlas: Any) -> float:
    """Every test atlas is the size of the adult mouse brain (brain scale 1):
    the fakes are a few planes, not brains; ``tests/test_brain_scale.py``
    measures real atlases and sets other sizes itself."""
    from langslice.core.atlas.size import REFERENCE_VOLUME_MM3

    return REFERENCE_VOLUME_MM3


PATCHES: tuple[tuple[str, str, Any], ...] = (
    ("langslice.core.deformation", "USE_PROCESS_POOL", False),
    ("langslice.core.deformation", "DETAIL_LEVEL", "coarse"),
    ("langslice.core.atlas.size", "brain_volume_mm3", _reference_brain),
)


def apply_patches() -> None:
    import importlib

    for module_name, attribute, value in PATCHES:
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            raise RuntimeError(f"{module_name}.{attribute} no longer exists; "
                               "update tests/synthetic_stack.py PATCHES")
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
                                     model=str(request.model), route="test-stub")

    resolved = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)
    return dataclasses.replace(resolved, call=generate)


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
    """The one JSON spelling snapshots are written and compared in."""
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def full_spec(folder: Path) -> Any:
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec, TransformSpec

    return JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none",
        tasks=["position", "transform", "nonlinear"],
        inputs={"pixel_size_um": PIXEL_SIZE_UM},
        position=PositionSpec(),
        transform=TransformSpec(angles=True),
        nonlinear=NonlinearSpec(provider="openai-oauth"),
    )

