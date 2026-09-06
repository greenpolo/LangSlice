"""Physical calibration: millimetres on the canvas, and the outlines drawn on it.

The alignment loop is a vision-action loop, so a render that lies about scale
compromises every decision made from it. These are the geometry pins: what one
millimetre is worth in pixels, how wide a known atlas lands, how long the scale
bar is, and where the outlines fall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.affine import normalized_physical_affine, physical_affine_matrix
from langslice.atlas.recolor import color_lut
from langslice.atlas.render import family_mapping, family_outlines
from langslice.image_prep import read_pixel_size_um
from langslice.linear.engine import build_context, ingest
from langslice.linear.render import (
    canvas_geometry,
    canvas_um_per_px,
    physical_overlay,
    scale_bar_px,
)
from langslice.linear.spec import JobSpec
from langslice.linear.transform import calibrate

#: The atlas fake below is 25 um per voxel, like allen_mouse_25um.
ATLAS_UM = 25.0
#: ... and its anatomy is exactly 40 voxels = 1.000 mm wide.
BLOB_VOXELS = 40


class TwoRegionAtlas:
    """A 25 um atlas whose anatomy is a 1 mm square with a second region in it."""

    atlas_name = "fake_two_region_25um"
    orientation = "asr"
    resolution = (ATLAS_UM, ATLAS_UM, ATLAS_UM)
    metadata = {"species": "mouse"}
    structures = {
        1: {"id": 1, "acronym": "A", "name": "A", "structure_id_path": [1],
            "rgb_triplet": [220, 40, 40]},
        2: {"id": 2, "acronym": "B", "name": "B", "structure_id_path": [1, 2],
            "rgb_triplet": [40, 60, 220]},
    }

    def __init__(self, n_slices: int = 20, size: int = 100):
        annotation = np.zeros((n_slices, size, size), dtype=np.int32)
        lo = (size - BLOB_VOXELS) // 2
        annotation[:, lo : lo + BLOB_VOXELS, lo : lo + BLOB_VOXELS] = 1
        annotation[:, lo + 10 : lo + 30, lo + 10 : lo + 30] = 2
        self.annotation = annotation
        self.template = (annotation > 0).astype(np.uint8) * 200


def _section(size: tuple[int, int] = (300, 300)) -> Image.Image:
    """A dark field with a bright square of "tissue" in the middle."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[size[1] // 2 - 40 : size[1] // 2 + 40, size[0] // 2 - 40 : size[0] // 2 + 40] = 200
    return Image.fromarray(arr, mode="RGB")


def _overlay(params: dict[str, float], um_per_px: float = 10.0, **kwargs) -> np.ndarray:
    image = physical_overlay(
        _section(), um_per_px, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, params,
        **kwargs,
    )
    return np.asarray(image.convert("RGB"))


_IDENTITY = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
}


# --- reading the pixel size off a file -----------------------------------


def test_pixel_size_comes_off_the_resolution_tags(tmp_path: Path):
    path = tmp_path / "tagged.tif"
    Image.new("RGB", (32, 32)).save(path, dpi=(300.0, 300.0))
    # 300 pixels per inch: 25400 um / 300
    assert read_pixel_size_um(path) == pytest.approx(25400.0 / 300.0, rel=1e-6)


def test_pixel_size_is_none_when_the_file_does_not_say(tmp_path: Path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (32, 32)).save(plain)
    assert read_pixel_size_um(plain) is None
    assert read_pixel_size_um(tmp_path / "missing.tif") is None


def test_ome_xml_pixel_size_wins_over_the_tags(tmp_path: Path):
    """An OME header states its own unit, and it beats the resolution tags."""
    path = tmp_path / "ome.tif"
    ome = (
        '<OME><Image><Pixels PhysicalSizeX="0.75488" PhysicalSizeXUnit="um" '
        '/></Image></OME>'
    )
    Image.new("RGB", (32, 32)).save(path, dpi=(300.0, 300.0), description=ome)
    assert read_pixel_size_um(path) == pytest.approx(0.75488, rel=1e-6)


def test_the_micro_sign_survives_a_mangled_encoding():
    """tifffile writes "µm"; its UTF-8 bytes reach us as "Âµm" often enough."""
    from langslice.image_prep import _unit_to_um

    assert _unit_to_um("Âµm") == 1.0
    assert _unit_to_um("µm") == 1.0
    assert _unit_to_um("um") == 1.0
    assert _unit_to_um("") == 1.0  # OME's default
    assert _unit_to_um("mm") == 1000.0
    assert _unit_to_um("furlong") is None


@pytest.mark.skipif(
    not Path("~/LSD_910/images/M04/M04_A_04_Overlay.tif").expanduser().exists(),
    reason="LSD_910 M04 is not on this machine",
)
def test_a_real_acquisition_reads_its_own_pixel_size():
    path = Path("~/LSD_910/images/M04/M04_A_04_Overlay.tif").expanduser()
    assert read_pixel_size_um(path) == pytest.approx(0.7549, abs=1e-4)


# --- millimetres on the canvas -------------------------------------------


def test_one_millimetre_is_a_thousand_micrometres_of_pixels():
    for um_per_px in (10.0, 0.75488, 25.0):
        matrix = physical_affine_matrix(
            rotation_deg=0.0, scale_x=1.0, scale_y=1.0,
            translate_x_mm=1.0, translate_y_mm=-2.0,
            size=(300, 200), um_per_px=um_per_px,
        )
        assert matrix[0, 2] == pytest.approx(1000.0 / um_per_px)
        assert matrix[1, 2] == pytest.approx(-2000.0 / um_per_px)


def test_a_one_millimetre_shift_moves_the_tissue_by_that_many_pixels():
    def centroid(rgb: np.ndarray) -> tuple[float, float]:
        # The tissue square only: the outlines, the bar and the caption do
        # not move with the section.
        ys, xs = np.nonzero((rgb == 200).all(axis=2))
        return float(xs.mean()), float(ys.mean())

    still = _overlay(_IDENTITY)
    moved = _overlay({**_IDENTITY, "translate_x_mm": 1.0})
    x0, y0 = centroid(still)
    x1, y1 = centroid(moved)
    assert x1 - x0 == pytest.approx(100.0, abs=2.0)  # 1 mm at 10 um/px
    assert y1 - y0 == pytest.approx(0.0, abs=2.0)


def test_the_normalized_numbers_describe_the_same_physical_map():
    params = {**_IDENTITY, "translate_x_mm": 0.5, "rotation_deg": 0.0}
    six = normalized_physical_affine(size=(300, 200), um_per_px=10.0, **params)
    assert six[2] == pytest.approx(50.0 / 300.0)  # half a millimetre of width


# --- the atlas at true scale ---------------------------------------------


def test_a_known_atlas_width_lands_at_the_pixel_width_it_should():
    geometry = canvas_geometry((300, 300), 10.0, TwoRegionAtlas(), 0.2, "coronal")
    assert geometry.atlas_scale == pytest.approx(ATLAS_UM / 10.0)

    outlines = family_outlines(TwoRegionAtlas(), 0.2, plane="coronal")
    points = np.concatenate([poly for _color, poly in outlines])
    drawn = (points[:, 0].max() - points[:, 0].min()) * geometry.atlas_scale
    # 40 voxels x 25 um = 1.000 mm, and 1 mm on a 10 um/px canvas is 100 px.
    # Contours run through pixel CENTRES, so the span is one voxel (2.5 px)
    # short of the region's outer edge.
    assert drawn == pytest.approx(100.0 - ATLAS_UM / 10.0, abs=1.0)


def test_the_canvas_grows_when_the_atlas_would_not_fit():
    small = canvas_geometry((40, 40), 10.0, TwoRegionAtlas(), 0.2, "coronal")
    assert small.size[0] >= 100 and small.size[1] >= 100
    assert small.section_offset[0] == (small.size[0] - 40) // 2
    fixed = canvas_geometry(
        (40, 40), 10.0, TwoRegionAtlas(), 0.2, "coronal", pad_to_fit_atlas=False
    )
    assert fixed.size == (40, 40)


def test_the_scale_bar_is_one_millimetre_long():
    assert scale_bar_px(10.0) == 100
    assert scale_bar_px(0.75488) == 1325

    rgb = _overlay(_IDENTITY)
    bar_rows = [
        row for row in range(rgb.shape[0] - 40, rgb.shape[0])
        if (rgb[row, :150] >= 230).all(axis=1).sum() >= 100
    ]
    assert bar_rows, "no 1 mm bar found in the bottom strip"
    run = (rgb[bar_rows[0]] >= 230).all(axis=1).sum()
    assert run == pytest.approx(101, abs=2)  # inclusive endpoints


def test_the_template_only_shows_when_it_is_asked_for():
    plain = _overlay(_IDENTITY)
    with_template = _overlay(_IDENTITY, show_template=True)
    # The atlas anatomy sits centred; the template lights it up under the lines.
    box = (slice(140, 160), slice(140, 160))
    assert with_template[box].mean() > plain[box].mean() + 5


# --- the outlines --------------------------------------------------------


def test_outlines_follow_the_family_boundaries_of_the_filled_map():
    from langslice.nonlinear.render import filled_regions

    atlas = TwoRegionAtlas()
    labels = atlas.annotation[0]
    mapping = family_mapping(np.unique(labels), atlas)
    assert len({mapping[1], mapping[2]}) == 2, "the two regions must not merge"

    families = np.zeros_like(labels)
    for uid, rep in mapping.items():
        families[labels == uid] = rep
    filled = filled_regions(families, lut=color_lut(atlas), smooth_window=9)

    # Every pixel whose neighbour is a different color is a family boundary.
    boundary = np.zeros(filled.shape[:2], dtype=bool)
    boundary[:, 1:] |= (filled[:, 1:] != filled[:, :-1]).any(axis=2)
    boundary[1:, :] |= (filled[1:, :] != filled[:-1, :]).any(axis=2)
    import cv2

    near = cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    points = np.concatenate(
        [poly for _color, poly in family_outlines(atlas, 0.2, plane="coronal")]
    )
    rows = np.clip(np.round(points[:, 1]).astype(int), 0, near.shape[0] - 1)
    cols = np.clip(np.round(points[:, 0]).astype(int), 0, near.shape[1] - 1)
    assert near[rows, cols].mean() > 0.98


def test_outlines_are_drawn_in_each_family_own_color():
    colors = {color for color, _poly in family_outlines(TwoRegionAtlas(), 0.2)}
    assert len(colors) == 2  # one per family, not one loud line color for all


# --- calibration in a run ------------------------------------------------


def _ctx(folder: Path, **inputs):
    spec = JobSpec(
        image_folder=str(folder),
        model="fake-model",
        preprocess="none",
        inputs=dict(inputs),
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: TwoRegionAtlas())
    return ctx, ingest(spec, ctx)


def test_calibration_reads_the_file_and_follows_the_downsample(tmp_path: Path):
    _section((1024, 1024)).save(tmp_path / "s.tif", dpi=(2540.0, 2540.0))  # 10 um/px
    ctx, state = _ctx(tmp_path)
    record = state.slices[0]

    assert ctx.calibration(record.id)[0] == pytest.approx(10.0)
    # The 512-long-edge render halves the resolution, so the canvas doubles.
    um, source = canvas_um_per_px(ctx, record, long_edge=512)
    assert source == "file"
    assert um == pytest.approx(20.0)


def test_the_host_pixel_size_overrides_the_file(tmp_path: Path):
    _section((512, 512)).save(tmp_path / "s.tif", dpi=(2540.0, 2540.0))
    ctx, state = _ctx(tmp_path, pixel_size_um=3.0)
    assert ctx.calibration(state.slices[0].id) == (3.0, "host")


def test_no_pixel_size_anywhere_is_estimated_never_fatal(tmp_path: Path):
    _section((512, 512)).save(tmp_path / "s.png")
    ctx, state = _ctx(tmp_path)
    record = state.slices[0]
    record.position_mm = 0.2
    assert canvas_um_per_px(ctx, record)[0] is None

    from langslice.linear.render import render_slice

    section = render_slice(ctx, record, long_edge=512)
    um, source = calibrate(state, ctx, record, section)
    assert source == "estimated"
    # The tissue square is 80 px of a 512 px section; the atlas anatomy is
    # 1 mm wide, so the estimate is around 1000/80 um per pixel.
    assert 8.0 < um < 16.0


def test_the_preview_tool_returns_one_image_and_the_numbers(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY
    from langslice.linear.transform import _build_align_tools

    _section((512, 512)).save(tmp_path / "s.tif", dpi=(2540.0, 2540.0))  # 10 um/px
    ctx, state = _ctx(tmp_path)
    record = state.slices[0]
    record.position_mm = 0.2
    box = _build_align_tools(state, ctx, record)
    preview = next(tool for tool in box.tools if tool.__name__ == "preview_transform")

    result = preview(0.0, 1.0, 1.0, 0.25, 0.0, True)
    assert result["status"] == "ok"
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1  # ONE image, not a panel strip
    assert result["params"]["translate_x_mm"] == 0.25
    assert result["calibration"]["source"] == "file"
    assert result["decomposition"]["rotation_deg"] == 0.0
    assert box.previews == 1


# --- the reasoning knob --------------------------------------------------


def test_reasoning_effort_reaches_a_model_that_has_one():
    from langslice.linear.session import build_agent
    from langslice.providers.openai_oauth import ChatGptLlm

    agent = build_agent(
        model=ChatGptLlm(model="gpt-5.6-sol"),
        name="t",
        instruction="i",
        tools=[],
        reasoning="high",
    )
    assert getattr(agent.model, "reasoning_effort", None) == "high"

    default = build_agent(
        model=ChatGptLlm(model="gpt-5.6-sol"), name="t", instruction="i", tools=[]
    )
    assert getattr(default.model, "reasoning_effort", None) == "medium"
    # A plain model string has no such knob and must survive untouched.
    assert build_agent(
        model="gemini-3-pro", name="t", instruction="i", tools=[], reasoning="high"
    ).model == "gemini-3-pro"
