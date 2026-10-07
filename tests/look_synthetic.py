"""A small stack on the synthetic atlas for the look, positioning and zoom tests.

Built from core classes only (a :class:`Workspace` and a :class:`StackState`):
three sections rendered from :class:`tests.deformable_synthetic.SyntheticAtlas`
at 25 um/px, plus a "marked" section: a gray slab with one white square at a
known place, so a zoom can be checked to land where it was asked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace
from tests.deformable_synthetic import SyntheticAtlas, bump_field, render_section

#: Section ids in stack order (file order), and their positions (mm).
SECTIONS = ("s0.png", "s1.png", "s2.png")
POSITIONS = (0.05, 0.15, 0.10)
MARKED = "marked.png"
#: The marked section: a slab of this size with a square at this box (x0, y0, x1, y1).
MARKED_SIZE = (320, 240)
SQUARE = (60, 50, 100, 90)


def _marked() -> Image.Image:
    width, height = MARKED_SIZE
    image = np.full((height, width, 3), 236, dtype=np.uint8)
    image[20:height - 20, 20:width - 20] = 120
    x0, y0, x1, y1 = SQUARE
    image[y0:y1, x0:x1] = 255
    return Image.fromarray(image)


def stack(tmp_path: Path, *, marked: bool = False) -> tuple[Workspace, StackState]:
    """``(workspace, state)``: the three synthetic sections (and with
    *marked*, the marked one last), each at its position."""
    folder = tmp_path / "images"
    folder.mkdir(exist_ok=True)
    atlas = SyntheticAtlas()
    for seed, name in enumerate(SECTIONS):
        image, _labels = render_section(atlas, bump_field([]), seed=seed)
        image.save(folder / name)
    names: list[str] = list(SECTIONS)
    positions: list[float] = list(POSITIONS)
    if marked:
        _marked().save(folder / MARKED)
        names.append(MARKED)
        positions.append(0.2)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": 25.0})
    ws = Workspace(spec=spec, image_folder=str(folder), emit=lambda _m: None,
                   atlas_loader=lambda _name: atlas)
    state = StackState(image_folder=str(folder), atlas=atlas.atlas_name, plane="coronal",
                       slices=[SliceState(id=name, index_original=k, index_corrected=k,
                                          position_mm=position)
                               for k, (name, position) in enumerate(zip(names, positions,
                                                                        strict=True))])
    return ws, state


def record(state: StackState, section_id: str) -> SliceState:
    """The section *section_id* of *state* (it must be there)."""
    found = state.by_id(section_id)
    assert found is not None, section_id
    return found
