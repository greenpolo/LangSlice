"""A section and its atlas drawn as separate panels are at ONE scale.

``stacked``, ``side_by_side``, ``view_stack``'s sheet and the opening strips
frame the section and the atlas apart; each pair is drawn at one micrometres
per pixel, so the brain's width in pixels on the two panels is in the
physical ratio (1 for a whole section at its true pixel size), never each
fitted to the picture.

The fixture is the golden run's synthetic atlas: a 7.0 x 5.2 mm ellipse
brain (50 um voxels), and sections rendered from it at 25 um/px, whole or
with the right third missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import atlas_fetch, pictures, placement, sheets
from langslice.core.display import default_options, framed_atlas
from langslice.core.opening import pair_tiles
from langslice.core.scale import (
    atlas_extent_um,
    brain_extent_um,
    pair_scale,
    section_at,
    stored_scale,
)
from langslice.core.spec import JobSpec
from langslice.job.job import ingest
from tests.deformable_synthetic import SECTION_SIZE, SyntheticAtlas, bump_field, render_section

#: The brain ellipse: 140 x 104 voxels at 50 um.
BRAIN_UM = (7000.0, 5200.0)
#: The fraction of the section's width that survives on the damaged one.
KEPT = 0.68
#: Pixel tolerance of a measured width (edges, resampling).
TOLERANCE_PX = 4


def _write(folder: Path) -> None:
    atlas = SyntheticAtlas()
    whole, _ = render_section(atlas, np.zeros((*SECTION_SIZE[::-1], 2)), seed=0)
    whole.save(folder / "whole.png")
    width, height = SECTION_SIZE
    remove = np.zeros((height, width), dtype=bool)
    remove[:, int(width * KEPT):] = True
    damaged, _ = render_section(atlas, bump_field([]), remove=remove, seed=1)
    damaged.save(folder / "damaged.png")


def _job(tmp_path: Path, pixel_size_um: float = 25.0) -> tuple[Any, Any]:
    folder = tmp_path / "images"
    folder.mkdir()
    _write(folder)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": pixel_size_um})
    atlas = SyntheticAtlas()
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for record in state.in_order():
        record.position_mm = 0.1
    return state, ctx


def _tissue_box(picture: Image.Image) -> tuple[int, int]:
    """``(width, height)`` in pixels of the stained tissue on a brightfield panel."""
    gray = np.asarray(picture.convert("L"), dtype=np.float32)
    mask = (gray > 40) & (gray < 205)  # the slide is ~236, a caption band black
    return _box(mask)


def _brain_box(picture: Image.Image) -> tuple[int, int]:
    """``(width, height)`` in pixels of the atlas anatomy on its black panel."""
    gray = np.asarray(picture.convert("L"), dtype=np.float32)
    return _box(gray > 25)


def _box(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


@pytest.fixture
def no_captions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pictures without their caption band, so a panel is only its content."""
    for module in (sheets, pictures, placement, atlas_fetch):
        monkeypatch.setattr(module, "caption", lambda image, _text: image)


def _split(picture: Image.Image, top_height: int) -> tuple[Image.Image, Image.Image]:
    """A :func:`langslice.core.sheets.stacked` picture back into its two panels."""
    return (picture.crop((0, 0, picture.width, top_height)),
            picture.crop((0, top_height + 6, picture.width, picture.height)))


def _assert_true_size(section: Image.Image, atlas: Image.Image, *, kept: float = 1.0,
                      ratio: float = 1.0) -> None:
    """The section's tissue against the atlas brain: the height in *ratio*,
    the width in *ratio* times the fraction *kept*."""
    tissue_w, tissue_h = _tissue_box(section)
    brain_w, brain_h = _brain_box(atlas)
    assert abs(tissue_h - ratio * brain_h) <= TOLERANCE_PX, (tissue_h, brain_h)
    assert abs(tissue_w - ratio * kept * brain_w) <= TOLERANCE_PX + 0.02 * brain_w, (
        tissue_w, brain_w)


# --- the opening strips -----------------------------------------------------


@pytest.mark.parametrize("name, kept", [("whole.png", 1.0), ("damaged.png", KEPT)])
def test_opening_tiles_draw_a_section_and_its_atlas_at_one_scale(
    tmp_path: Path, name: str, kept: float,
):
    state, ctx = _job(tmp_path)
    section, atlas = pair_tiles(ctx, state, state.by_id(name), 0.1, 256)
    assert max(*section.size, *atlas.size) <= 256
    _assert_true_size(section, atlas, kept=kept)
    # The atlas, the larger panel here, fills the tile.
    assert max(atlas.size) >= 254


def test_a_wrong_pixel_size_shows_as_a_wrong_size(tmp_path: Path):
    """A section declared at twice its true pixel size is drawn twice as
    large as the atlas: the panels show the calibration, they do not hide it."""
    state, ctx = _job(tmp_path, pixel_size_um=50.0)
    section, atlas = pair_tiles(ctx, state, state.by_id("whole.png"), 0.1, 256)
    _assert_true_size(section, atlas, ratio=2.0)
    assert max(section.size) >= 254  # now the section is the larger panel


def test_a_stored_transform_scale_is_drawn_as_the_overlay_draws_it(tmp_path: Path):
    state, ctx = _job(tmp_path)
    record = state.by_id("whole.png")
    record.transform = {"params": [0.8, 0.0, 0.1, 0.0, 0.8, 0.1], "kind": "stored"}
    assert stored_scale(record) == pytest.approx(0.8)
    section, atlas = pair_tiles(ctx, state, record, 0.1, 256)
    _assert_true_size(section, atlas, ratio=0.8)


# --- view_placement / set_positions: stacked and side_by_side -----------------


@pytest.mark.parametrize("name, kept", [("whole.png", 1.0), ("damaged.png", KEPT)])
def test_stacked_draws_the_section_over_the_atlas_at_one_scale(
    tmp_path: Path, no_captions: None, name: str, kept: float,
):
    state, ctx = _job(tmp_path)
    record = state.by_id(name)
    options = default_options("stacked", long_edge=512)
    placed = placement.placement_pictures(ctx, state, record, 0.1, options, {})
    (picture,) = placed.images
    shown, working = pair_scale(ctx, state, record, 0.1, 512)
    top = section_at(ctx, record, shown, working_um=working, long_edge=512)
    section, atlas = _split(picture, top.height)
    _assert_true_size(section, atlas, kept=kept)
    assert max(picture.width, top.height, picture.height - top.height - 6) <= 512


@pytest.mark.parametrize("name, kept", [("whole.png", 1.0), ("damaged.png", KEPT)])
def test_side_by_side_draws_both_references_at_one_scale(
    tmp_path: Path, no_captions: None, name: str, kept: float,
):
    state, ctx = _job(tmp_path)
    record = state.by_id(name)
    options = default_options("side_by_side", atlas_channels=("template",), long_edge=512)
    working: placement.Working = {}
    first = placement.placement_pictures(ctx, state, record, 0.1, options, working)
    section, atlas = first.images
    _assert_true_size(section, atlas, kept=kept)
    assert max(*section.size, *atlas.size) <= 512
    # The door shows a section once per call: its picture is the same at
    # every position, so it reads true against each atlas.
    other = placement.placement_pictures(ctx, state, record, 0.2, options, working)
    assert other.images[0] is section
    _assert_true_size(section, other.images[1], kept=kept)


def test_side_by_side_with_lines_keeps_the_scale(tmp_path: Path, no_captions: None):
    state, ctx = _job(tmp_path)
    record = state.by_id("damaged.png")
    options = default_options("side_by_side", long_edge=512)  # template + borders
    section, atlas = placement.placement_pictures(ctx, state, record, 0.1, options, {}).images
    _assert_true_size(section, atlas, kept=KEPT)


# --- view_stack --------------------------------------------------------------


def test_the_contact_sheet_draws_every_pair_at_one_scale(tmp_path: Path, no_captions: None):
    """A damaged section is drawn at its true size over its atlas, not
    magnified to the tile like a whole one."""
    state, ctx = _job(tmp_path)
    options = default_options("stacked")
    given: dict[str, float] = {}

    def under(record: Any, um_per_px: float) -> Image.Image:
        given[record.id] = um_per_px
        return framed_atlas(ctx, state, float(record.position_mm), options,
                            um_per_px=um_per_px, angles=record.angles)

    tiles = dict(sheets.stack_pictures(state, ctx, long_edge=256, by_position=True,
                                       under=under))
    tissue: dict[str, tuple[int, int]] = {}
    for label, picture in tiles.items():
        record = next(r for r in state.in_order() if r.id in label)
        shown, working = pair_scale(ctx, state, record, 0.1, 256)
        assert given[record.id] == pytest.approx(shown)
        top = section_at(ctx, record, shown, working_um=working, long_edge=256)
        section, atlas = _split(picture, top.height)
        _assert_true_size(section, atlas, kept=1.0 if record.id == "whole.png" else KEPT)
        tissue[record.id] = _tissue_box(section)
    # Both pairs are atlas-bound at the same position: one scale for both, so
    # the two sections share their height and the damaged one is narrower.
    assert abs(tissue["whole.png"][1] - tissue["damaged.png"][1]) <= TOLERANCE_PX
    assert tissue["damaged.png"][0] < 0.8 * tissue["whole.png"][0]


# --- the bound side_by_side sizes by ---------------------------------------------


@pytest.mark.parametrize("angles", [(0.0, 0.0), (3.0, -2.0)])
def test_the_brain_bound_holds_every_framed_atlas_plane(tmp_path: Path, angles: Any):
    state, ctx = _job(tmp_path)
    bound = brain_extent_um(ctx, state, angles=angles)
    largest = max(atlas_extent_um(ctx, state, position, angles=angles)
                  for position in (0.05, 0.1, 0.15, 0.2))
    assert largest <= bound
    if angles == (0.0, 0.0):
        assert bound <= 1.2 * max(BRAIN_UM) * 1.12  # a bound, not a guess
