"""The opening images: ABBA-style strips of sections over their atlas.

Strip long edge = the model lane's largest image, tiles = the level's opening
size, every strip inside the lane's patch budget, the atlas row only with
positions, the atlas reference only when a section has none.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.opening import (
    CLAUDE_IMAGE_LIMIT,
    CLAUDE_MAX_IMAGE_EDGE,
    CLAUDE_MAX_IMAGE_PATCHES,
    COLUMN_GAP,
    OPENAI_MAX_IMAGE_EDGE,
    OPENAI_MAX_IMAGE_PATCHES,
    pack_strips,
    patches,
    strip_layout,
    tile_label,
)
from langslice.core.sizes import PICTURE_EDGES
from langslice.core.spec import JobSpec
from langslice.doors.tools.media import opening_parts
from langslice.doors.tools.view_options import image_limit
from langslice.job.job import ingest
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _stack(folder: Path, n: int, *, size: tuple[int, int] = (160, 120),
           level: str = "low", model: str = "openai-oauth/gpt-6-astra") -> tuple[Any, Any]:
    folder.mkdir()
    width, height = size
    for index in range(n):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[height // 8: height * 7 // 8, width // 8: width * 7 // 8] = 60 + index
        Image.fromarray(arr).save(folder / f"s{index:02d}.png")
    spec = JobSpec(image_folder=str(folder), model=model, preprocess="none",
                   image_resolution=level)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    return ingest(spec, ctx), ctx


def _strips(parts: list[Any], prefix: str = "Strip ") -> list[tuple[str, Image.Image]]:
    """``(text, image)`` for every strip whose text starts with *prefix*."""
    out = []
    for index, part in enumerate(parts):
        if part.text and part.text.startswith(prefix):
            out.append((part.text, Image.open(io.BytesIO(parts[index + 1].inline_data.data))))
    return out


def test_the_lane_limits_are_the_verified_numbers():
    assert (OPENAI_MAX_IMAGE_EDGE, OPENAI_MAX_IMAGE_PATCHES) == (2048, 2500)
    assert CLAUDE_MAX_IMAGE_EDGE == 1568
    assert CLAUDE_IMAGE_LIMIT == (1568, CLAUDE_MAX_IMAGE_PATCHES)

    class Ctx:
        model = "openai-oauth/gpt-6-astra"

    assert image_limit(Ctx()) == (2048, 2500)
    Ctx.model = "fake-model"
    assert image_limit(Ctx()) == (2048, 2500)


@pytest.mark.parametrize(("level", "per_strip"), [("low", 8), ("medium", 5), ("high", 4)])
def test_tiles_per_strip_fill_the_long_edge(level: str, per_strip: int):
    opening = PICTURE_EDGES[level][0]
    count, tile = strip_layout(OPENAI_MAX_IMAGE_EDGE, opening)
    assert count == per_strip
    assert opening - 8 <= tile <= opening
    assert count * tile + (count - 1) * COLUMN_GAP <= OPENAI_MAX_IMAGE_EDGE
    assert (count + 1) * tile > OPENAI_MAX_IMAGE_EDGE


@pytest.mark.parametrize(("level", "strips"), [("low", 5), ("medium", 8), ("high", 10)])
def test_a_40_section_stack_takes_more_strips_at_higher_levels(
    tmp_path: Path, level: str, strips: int,
):
    state, ctx = _stack(tmp_path / level, 40, level=level)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 1.0 + 0.1 * index
    parts = opening_parts(state, ctx)
    found = _strips(parts)
    assert len(found) == strips
    count, tile = strip_layout(OPENAI_MAX_IMAGE_EDGE, PICTURE_EDGES[level][0])
    for _text, image in found:
        assert image.width <= OPENAI_MAX_IMAGE_EDGE
        assert patches(image.size) <= OPENAI_MAX_IMAGE_PATCHES
    # Every full strip is the strip's long edge to within one tile.
    assert all(image.width > OPENAI_MAX_IMAGE_EDGE - tile for _text, image in found[:-1])
    # Ordered, labelled, every section once.
    named = [name for text, _ in found for name in text.split(": ", 1)[1].split(", ")]
    assert named == [f"{r.index_corrected}: {r.id}" for r in state.in_order()]
    # All positioned: no atlas reference.
    assert not any(p.text and p.text.startswith("Atlas reference") for p in parts)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_tall_paired_strips_stay_inside_the_patch_budget(tmp_path: Path, level: str):
    """Portrait sections at full size over their atlas: the worst strip height."""
    state, ctx = _stack(tmp_path / level, 10, size=(1200, 1800), level=level)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 1.0 + 0.2 * index
        record.damage_marked = True  # a longer label
    for limit in (None, CLAUDE_IMAGE_LIMIT):
        edge, budget = limit or (OPENAI_MAX_IMAGE_EDGE, OPENAI_MAX_IMAGE_PATCHES)
        for _text, image in _strips(opening_parts(state, ctx, limit=limit)):
            assert max(image.size) <= edge
            assert patches(image.size) <= budget


def test_the_atlas_row_appears_only_with_positions(tmp_path: Path):
    state, ctx = _stack(tmp_path / "s", 3)
    bare = opening_parts(state, ctx)
    assert "Beneath each section" not in (bare[0].text or "")
    assert any(p.text and p.text.startswith("Atlas reference strip") for p in bare)
    ((_, unplaced),) = _strips(bare)

    state.in_order()[0].position_mm = 2.0
    mixed = opening_parts(state, ctx)
    assert "Beneath each section" in (mixed[0].text or "")
    # A section without a position: the reference is still sent.
    assert any(p.text and p.text.startswith("Atlas reference strip") for p in mixed)
    ((_, paired),) = _strips(mixed)
    assert paired.height > unplaced.height and paired.width == unplaced.width

    for record in state.in_order():
        record.position_mm = 2.0
    placed = opening_parts(state, ctx)
    assert not any(p.text and p.text.startswith("Atlas") for p in placed)
    assert sum(p.inline_data is not None for p in placed) == 1


def test_the_reference_is_atlas_strips_at_the_tile_size(tmp_path: Path):
    state, ctx = _stack(tmp_path / "s", 2)
    parts = opening_parts(state, ctx)
    reference = next(p.text for p in parts if p.text and p.text.startswith("Atlas reference"))
    count = int(reference.split("the atlas at ")[1].split()[0])
    atlas = _strips(parts, "Atlas strip ")
    per_strip, _tile = strip_layout(OPENAI_MAX_IMAGE_EDGE, PICTURE_EDGES["low"][0])
    assert len(atlas) == -(-count // per_strip)
    assert all(image.width <= OPENAI_MAX_IMAGE_EDGE for _text, image in atlas)


def test_tile_labels_name_index_file_and_short_flags(tmp_path: Path):
    state, _ctx = _stack(tmp_path / "s", 2)
    record = state.in_order()[1]
    assert tile_label(record) == "1: s01.png"
    record.rotation_deg, record.flip, record.damage_marked = 90, True, True
    record.damage_note = "a long note about the tear that belongs in the status table"
    assert tile_label(record) == "1: s01.png [rot 90, flipped, damaged]"


def test_a_column_past_the_budget_starts_the_next_strip():
    tall = [Image.new("RGB", (100, 900))] * 2
    columns = [tall for _ in range(5)]
    strips = pack_strips(columns, 100, 5, budget=patches((210, 1804)))
    assert [start for start, _ in strips] == [0, 2, 4]
    # A single column always makes a strip, whatever it costs.
    assert len(pack_strips(columns[:1], 100, 5, budget=1)) == 1
