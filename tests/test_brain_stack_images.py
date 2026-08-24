"""The stack as the agent steps see it: one labelled image per section.

Every agent step seeds its session with ``stack_image_parts`` — a text label
followed by that section's own image, in corrected order — instead of one
thumbnail grid. These tests pin the ordering, the label/image pairing, the
corrected view (flips are mirrored), the render size, and the render cache
that keeps 40-section stacks from being prepared over and over.
"""

import asyncio
import io
from pathlib import Path

import numpy as np
from PIL import Image

from langslice.linear.whole_brain import _step_common
from langslice.linear.whole_brain._step_common import (
    SEED_IMAGE_LONG_EDGE,
    render_slice,
    stack_image_parts,
)
from langslice.linear.whole_brain.engine import EngineContext, build_context
from langslice.linear.whole_brain.nodes import ingest
from langslice.linear.whole_brain.position import build_position_seed_message
from langslice.linear.whole_brain.review import build_review_seed_message
from langslice.linear.whole_brain.state import BrainConfig, StackState
from langslice.linear.whole_brain.survey import build_seed_message


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    template = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


def _marked_section(size: tuple[int, int] = (120, 90)) -> Image.Image:
    """Dim tissue inset in a dark field, bright block on its LEFT — a mirror marker.

    Shaped like a real section rather than a half-and-half canvas: the
    whole-brain visual path frames sections to their tissue before sending
    them, and an image whose two halves are equally plausible as foreground
    has no bounding box to find.
    """
    width, height = size
    arr = np.full((height, width, 3), 8, dtype=np.uint8)
    x0, x1 = width // 6, width - width // 6
    y0, y1 = height // 6, height - height // 6
    arr[y0:y1, x0:x1] = 60
    arr[y0:y1, x0 : x0 + (x1 - x0) // 3] = 250
    return Image.fromarray(arr, mode="RGB")


def _small_tissue_section(size: tuple[int, int] = (240, 180)) -> Image.Image:
    """A section adrift in its scan: tissue covers about a tenth of the frame."""
    width, height = size
    arr = np.full((height, width, 3), 235, dtype=np.uint8)
    arr[30:80, 40:120] = 25
    return Image.fromarray(arr, mode="RGB")


def _tissue_fill(image: Image.Image, background: int = 235) -> float:
    arr = np.asarray(image.convert("L"), dtype=np.int16)
    return float((np.abs(arr - background) > 30).mean())


def _stack(
    folder: Path, n: int = 3, *, section=_marked_section, **kwargs
) -> tuple[StackState, EngineContext]:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        section().save(folder / f"slice_{index:02d}.png")
    ctx = build_context(
        BrainConfig(image_folder=str(folder), preprocess="none", **kwargs),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    return state, ctx


def _decode(part) -> Image.Image:
    assert part.inline_data is not None
    assert part.inline_data.data is not None
    return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")


def _image_parts(parts) -> list:
    return [part for part in parts if part.inline_data is not None]


def _halves(image: Image.Image) -> tuple[float, float]:
    arr = np.asarray(image, dtype=np.float64)
    middle = arr.shape[1] // 2
    return float(arr[:, :middle].mean()), float(arr[:, middle:].mean())


# --- stack_image_parts ---------------------------------------------------


def test_every_section_gets_a_label_then_its_own_image_in_corrected_order(
    tmp_path: Path,
):
    state, ctx = _stack(tmp_path, n=4)
    # Corrected order is not discovery order, and flags ride on the label.
    for record, index in zip(state.in_order(), [3, 2, 1, 0], strict=True):
        record.index_corrected = index
    state.by_id("slice_00.png").flip = True  # type: ignore[union-attr]
    damaged = state.by_id("slice_01.png")
    assert damaged is not None
    damaged.damaged = True
    damaged.damage_note = "missing hemisphere"

    parts = stack_image_parts(state, ctx)

    # One intro text part, then a (label, image) pair per section.
    assert len(parts) == 1 + 2 * len(state.slices)
    labels = [part.text for part in parts[1::2]]
    assert [part.inline_data is not None for part in parts[2::2]] == [True] * 4
    assert labels == [
        "0: slice_03.png",
        "1: slice_02.png",
        "2: slice_01.png  [damaged: missing hemisphere]",
        "3: slice_00.png  [flipped]",
    ]


def test_a_flipped_section_is_sent_mirrored(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=2)

    upright = _decode(_image_parts(stack_image_parts(state, ctx))[0])
    state.in_order()[0].flip = True
    mirrored = _decode(_image_parts(stack_image_parts(state, ctx))[0])

    left, right = _halves(upright)
    assert left > right * 2  # the marker starts on the left
    flipped_left, flipped_right = _halves(mirrored)
    assert flipped_right > flipped_left * 2  # ...and lands on the right


def test_long_edge_bounds_the_image_the_model_gets(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=1)

    default = _decode(_image_parts(stack_image_parts(state, ctx))[0])
    small = _decode(_image_parts(stack_image_parts(state, ctx, long_edge=48))[0])

    # The synthetic section is smaller than the default, so it is not upscaled.
    assert max(default.size) <= SEED_IMAGE_LONG_EDGE
    assert max(small.size) == 48


# --- framing -------------------------------------------------------------


def test_sections_reach_the_model_framed_to_their_tissue(tmp_path: Path):
    """Apparent scale is a cue: a section adrift in its scan is cropped first."""
    state, ctx = _stack(tmp_path, n=1, section=_small_tissue_section)
    record = state.in_order()[0]

    framed = _decode(_image_parts(stack_image_parts(state, ctx))[0])
    unframed = render_slice(ctx, record, long_edge=SEED_IMAGE_LONG_EDGE)

    assert _tissue_fill(unframed) < 0.15
    assert _tissue_fill(framed) > 0.6


def test_view_slices_frames_the_sections_it_zooms_into(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=1, section=_small_tissue_section)
    view_slices = _step_common.make_view_slices(state, ctx)

    result = view_slices(["slice_00.png"])

    image = _decode(result[_step_common.TOOL_MEDIA_PARTS_KEY][0])
    assert _tissue_fill(image) > 0.6


def test_the_geometry_path_is_not_framed(tmp_path: Path):
    """The transforms step fits an affine against this render; cropping it
    would silently move the coordinate frame the affine is normalized in."""
    state, ctx = _stack(tmp_path, n=1, section=_small_tissue_section)
    record = state.in_order()[0]

    plain = render_slice(ctx, record, long_edge=SEED_IMAGE_LONG_EDGE)
    framed = render_slice(ctx, record, long_edge=SEED_IMAGE_LONG_EDGE, frame=True)

    assert plain.size == _small_tissue_section().size  # untouched framing
    assert framed.size != plain.size
    assert plain is not framed  # different cache keys, not one shared render


# --- seed messages -------------------------------------------------------


def test_every_seed_message_carries_the_sections_not_a_contact_sheet(tmp_path: Path):
    state, ctx = _stack(tmp_path, n=5)
    for record in state.in_order():
        record.position_mm = 1.0 + 0.2 * record.index_corrected
        record.position_source = "refined"
    assert state.contact_sheet  # ingest still writes the human-facing artifact

    for message in (
        build_seed_message(state, ctx),
        build_position_seed_message(state, ctx),
        build_review_seed_message(state, ctx),
    ):
        parts = message.parts or []
        images = _image_parts(parts)
        # One image per section and not one more: the sheet is not attached.
        assert len(images) == len(state.slices)
        text = "\n".join(part.text or "" for part in parts)
        assert "contact sheet" not in text.lower()
        for record in state.in_order():
            assert f"{record.index_corrected}: {record.id}" in text


# --- the render cache ----------------------------------------------------


def test_a_section_is_prepared_once_per_key_and_reused(tmp_path: Path, monkeypatch):
    state, ctx = _stack(tmp_path, n=1)
    record = state.in_order()[0]
    ctx.render_cache.clear()  # ingest's contact sheet already warmed 256 px

    prepared: list[int] = []
    real = _step_common.prepare_image_for_vlm

    def counting(image, *, max_long_edge: int):
        prepared.append(max_long_edge)
        return real(image, max_long_edge=max_long_edge)

    monkeypatch.setattr(_step_common, "prepare_image_for_vlm", counting)

    first = render_slice(ctx, record, long_edge=64)
    second = render_slice(ctx, record, long_edge=64)

    assert prepared == [64]
    assert second is first  # served straight from the cache

    # A flip is a different key, so it re-renders rather than serving a stale
    # (unmirrored) image.
    record.flip = True
    flipped = render_slice(ctx, record, long_edge=64)

    assert prepared == [64, 64]
    assert flipped is not first
    assert len(ctx.render_cache) == 2

    # ...and a different size is a different key too.
    render_slice(ctx, record, long_edge=32)
    assert prepared == [64, 64, 32]
