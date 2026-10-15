"""The image model's canvas shows the section upright, and its lines still read back exactly.

A section placed at a tilt (the synthetic atlas turned 30 degrees, and mirrored)
is drawn on the canvas with the atlas's dorsal direction straight up and its
midline down the canvas's middle; the rough borders drawn there, extracted as
a model's lines would be and read back through the request's
``atlas_to_canvas`` (``core.deformation.traced_lines``), land on the atlas
borders placed directly on the section.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.core import deformation, handoff
from langslice.core.atlas.render import annotation_slice, placed_border_coverage
from langslice.core.handoff import LinearRegistrationInput
from langslice.core.nonlinear import registration_tool as tool
from langslice.core.nonlinear.upright import upright
from langslice.core.oblique import plane_index_affine
from langslice.core.space import atlas_space_context
from langslice.core.state import SliceState, StackState
from langslice.providers.registry import ImageModel, resolve_image_model
from tests.deformable_synthetic import SyntheticAtlas

POSITION_MM = 0.1
#: About the canvas's own size, so a canvas pixel is about a section pixel.
RENDER_SIZE = (1260, 1080)


def _tilted(turn_deg: float, *, mirror: bool = False, scale: float = 5.7) -> np.ndarray:
    """Native atlas plane (160 x 120) -> a RENDER_SIZE section, turned *turn_deg*
    clockwise on the screen about its centre (and mirrored left-right first)."""
    phi = math.radians(turn_deg)
    turn = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    linear = scale * turn @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))
    atlas_centre = np.array([79.5, 59.5])
    shift = np.array([RENDER_SIZE[0] / 2 + 36.0, RENDER_SIZE[1] / 2 - 21.0]) - linear @ atlas_centre
    return np.array([[*linear[0], shift[0]], [*linear[1], shift[1]], [0.0, 0.0, 1.0]])


def _render(atlas: SyntheticAtlas, atlas_to_slice: np.ndarray) -> Image.Image:
    """The atlas template placed on a mid-grey slide, as a section photograph."""
    template = np.asarray(atlas.template[0], dtype=np.float32)
    placed = cv2.warpAffine(template, atlas_to_slice[:2], RENDER_SIZE, flags=cv2.INTER_LINEAR,
                            borderValue=0.0)
    grey = np.clip(60.0 + placed * 0.6, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([grey] * 3, axis=-1))


def _midline_on_canvas(atlas: SyntheticAtlas, placement: np.ndarray) -> np.ndarray:
    """Two native points of the atlas midline (dorsal first) on the canvas."""
    index = plane_index_affine(atlas, POSITION_MM, "coronal")
    context = atlas_space_context(atlas)
    ml = context.ml_axis_index
    split = round(context.shape[ml] / 2) - 0.5
    points = []
    for row in (10.0, 110.0):  # row 10 is dorsal on the synthetic plane
        col = (split - index[ml, 2] - index[ml, 0] * row) / index[ml, 1]
        points.append(placement @ np.array([col, row, 1.0]))
    return np.array(points)[:, :2]


@pytest.mark.parametrize(("turn_deg", "mirror"), [(30.0, False), (-38.0, True), (172.0, False)])
def test_the_canvas_midline_is_vertical_centred_and_dorsal_up(turn_deg, mirror):
    atlas = SyntheticAtlas()
    atlas_to_slice = _tilted(turn_deg, mirror=mirror)
    prepared = LinearRegistrationInput(_render(atlas, atlas_to_slice), atlas_to_slice,
                                       atlas.atlas_name, POSITION_MM, "coronal", 0.0, 0.0, {})
    canvas, _, placement, stance, render_to_canvas = tool._canvas_and_placement(
        prepared, SimpleNamespace(atlas=atlas), "openai-oauth", None)
    (top_x, top_y), (bottom_x, bottom_y) = _midline_on_canvas(atlas, placement)
    centre_x = (canvas.width - 1) / 2.0
    assert top_x == pytest.approx(centre_x, abs=1e-6)
    assert bottom_x == pytest.approx(centre_x, abs=1e-6)
    assert top_y < bottom_y  # dorsal up
    # The section's own turn undone; a mirror kept (a turn never changes handedness).
    assert stance.turn_deg == pytest.approx(-turn_deg, abs=1e-6)
    assert np.sign(np.linalg.det(placement[:2, :2])) == (-1.0 if mirror else 1.0)
    # The provider's frame (the fixed Codex-lane budget), cropped to the placed
    # atlas plus 15% of its box's larger side on every side, symmetric about the
    # midline: the footprint box with that margin fits, and fills the frame.
    assert canvas.width * canvas.height == pytest.approx(1024 * 1536, rel=2e-3)
    outline = tool.footprint(np.asarray(annotation_slice(atlas, POSITION_MM, plane="coronal")),
                             placement)
    (x0, y0), (x1, y1) = outline.min(axis=0) + 0.5, outline.max(axis=0) + 0.5
    margin = 0.15 * max(x1 - x0, y1 - y0)
    assert x0 - margin >= -1e-6 and x1 + margin <= canvas.width + 1e-6
    assert y0 - margin >= -1e-6 and y1 + margin <= canvas.height + 1e-6
    half = max(canvas.width / 2 - (x0 - margin), x1 + margin - canvas.width / 2)
    assert max(2 * half / canvas.width, (y1 - y0 + 2 * margin) / canvas.height) > 0.99
    # The padding is the slide's background colour.
    assert canvas.getpixel((0, 0)) == (60, 60, 60)


def test_a_sagittal_canvas_points_the_dorsal_direction_up():
    atlas = SyntheticAtlas()
    atlas_to_slice = _tilted(25.0)
    stance = upright(atlas, "sagittal", 0.1, 0.0, 0.0, atlas_to_slice)
    assert stance.through is None
    index = plane_index_affine(atlas, 0.1, "sagittal")
    dv = atlas_space_context(atlas).dv_axis_index  # index 0 is superior ("asr")
    dorsal_native = -np.array([index[dv, 1], index[dv, 0]])
    on_canvas = stance.rotation() @ atlas_to_slice[:2, :2] @ dorsal_native
    assert on_canvas[0] == pytest.approx(0.0, abs=1e-9) and on_canvas[1] < 0


def test_lines_drawn_on_the_turned_canvas_read_back_onto_the_section_borders(
        tmp_path: Path, monkeypatch):
    """The whole route: the canvas and rough overlay ``start_correction`` makes,
    a model that hands the rough overlay back, the extracted lines, and
    ``traced_lines`` carrying them back to the section through the request's
    matrix. Every read-back line pixel lies on the section's placed borders."""
    atlas = SyntheticAtlas()
    atlas_to_slice = _tilted(33.0, mirror=True)
    render = _render(atlas, atlas_to_slice)
    prepared = LinearRegistrationInput(render, atlas_to_slice, atlas.atlas_name, POSITION_MM,
                                       "coronal", 0.0, 0.0, {})
    monkeypatch.setattr(tool, "prepare_linear_registration", lambda *a, **k: prepared)
    monkeypatch.setattr(tool, "correction_fingerprint", lambda *a, **k: "geometry")
    monkeypatch.setattr(handoff, "correction_fingerprint", lambda *a, **k: "geometry")
    record = SliceState(id="s.png", index_original=0, index_corrected=0,
                        position_mm=POSITION_MM, transform={"params": [1, 0, 0, 0, 1, 0]})
    state = StackState(atlas=atlas.atlas_name, slices=[record])
    ctx = SimpleNamespace(atlas=atlas)

    def hand_back(request):  # the GPT twin: Image 2 is the placed borders
        return SimpleNamespace(image=request.reference_images[0], route="test")

    model = ImageModel("openai-oauth", resolve_image_model("openai-oauth").model, hand_back)
    started, job = tool.start_correction(state, ctx, record.id, image_model=model,  # type: ignore[arg-type]
                                         calls_dir=tmp_path / "calls")
    assert job is not None
    result = job()
    assert result["status"] == "ok" and result["model_border_pixels"] > 0
    record.image_correction = result

    grid = SimpleNamespace(record=record, image=render,
                           placement=SimpleNamespace(atlas_to_section=atlas_to_slice))
    lines, _ = deformation.traced_lines(state, ctx, grid)  # type: ignore[arg-type]
    labels = np.asarray(annotation_slice(atlas, POSITION_MM, plane="coronal"))
    shown = tool.shown_labels(atlas, labels, prepared)
    truth = placed_border_coverage(shown, atlas_to_slice, render.size, width_px=1.0) > 0.25
    distance = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 5)
    # Every read-back line pixel is on the placed borders (a band about two
    # section pixels wide), and every border pixel is within a pixel or so of a line.
    assert distance[lines].max() <= 1.0
    missed = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 5)[truth]
    assert np.percentile(missed, 99) <= 1.5 and missed.max() <= 2.5
