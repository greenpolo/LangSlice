"""`fit_affine` include/exclude: the silhouette fit on the kept atlas regions only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from tests.fakes import EllipseAtlas, ellipse_section

LEFT, RIGHT, CORE = 2, 3, 4

#: The whole-outline fit on the ellipse fixture, as computed by the code before
#: regions existed: regions must not move it.
PINNED_IOU = 0.985
PINNED_PHYSICAL = {"rotation_deg": -0.004, "scale_x": 0.9998, "scale_y": 1.1398,
                   "translate_x_mm": -0.2817, "translate_y_mm": -0.041, "shear": 0.0001,
                   "pivot": [0.5, 0.5]}
PINNED_PARAMS = [0.999848764400981, -1.8221567469621935e-05, -0.002050823046910042,
                 8.817667081387976e-05, 1.1397842132941491, -0.07076133781644516]


class HalvesAtlas:
    """One ellipse of brain split into a left and a right half, 100 um voxels.

    A small ``CORE`` region sits deep inside the left half, away from the
    outline.
    """

    atlas_name = "fake_halves_100um"
    orientation = "asr"
    resolution = (100.0, 100.0, 100.0)
    metadata = {"species": "mouse"}

    def __init__(self) -> None:
        plane = np.zeros((96, 128), dtype=np.int32)
        cv2.ellipse(plane, (64, 48), (46, 30), 0, 0, 360, LEFT, -1)
        plane[:, 64:][plane[:, 64:] == LEFT] = RIGHT
        cv2.circle(plane, (40, 48), 3, CORE, -1)
        self.annotation = np.repeat(plane[None], 20, axis=0)
        self.template = (self.annotation > 0).astype(np.uint8) * 200
        paths = {1: [1], LEFT: [1, LEFT], RIGHT: [1, RIGHT], CORE: [1, LEFT, CORE]}
        names = {1: "root", LEFT: "L", RIGHT: "R", CORE: "C"}
        self.structures = {
            uid: {"id": uid, "acronym": names[uid], "name": names[uid],
                  "structure_id_path": paths[uid], "rgb_triplet": [200, 200, 200]}
            for uid in paths
        }


def _left_half_section() -> Image.Image:
    """The atlas's LEFT half only, at 50 um/px (twice the atlas's pixels)."""
    canvas = np.full((160, 200, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (100, 80), (92, 60), 0, 0, 360, (30, 30, 30), -1)
    canvas[:, 100:] = 240
    return Image.fromarray(canvas, mode="RGB")


def _box(folder: Path, atlas: Any, image: Image.Image, *, pixel_size_um: float | None = None,
         position_mm: float = 10.0):
    image.save(folder / "s0.png")
    inputs = {"pixel_size_um": pixel_size_um} if pixel_size_um else {}
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   inputs=inputs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    state.slices[0].position_mm = position_mm
    return state, build_tools(state, ctx, spec)


def _fit(box: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return next(t for t in box.tools if t.__name__ == "fit_affine")(*args, **kwargs)


def test_the_default_fit_is_unchanged(tmp_path: Path):
    """No regions: the whole-outline fit, numbers pinned from before regions existed."""
    state, box = _box(tmp_path, EllipseAtlas(), ellipse_section())
    result = _fit(box, [], "silhouette")
    row = result["results"][0]
    assert "regions" not in row
    assert row["iou"] == PINNED_IOU
    assert row["physical"] == PINNED_PHYSICAL
    assert state.slices[0].transform["params"] == pytest.approx(PINNED_PARAMS, abs=1e-12)
    assert "regions" not in state.slices[0].transform


def test_exclude_fits_the_kept_half_at_true_scale(tmp_path: Path):
    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    whole = _fit(box, ["s0.png"], "silhouette")["results"][0]
    _fit_undo = next(t for t in box.tools if t.__name__ == "undo")
    _fit_undo()
    # Against the whole outline the half section is turned onto the long axis.
    assert abs(whole["physical"]["rotation_deg"]) > 45 and whole["iou"] < 0.9
    kept = _fit(box, ["s0.png"], "silhouette", exclude=["R"])
    row = kept["results"][0]
    assert row["status"] == "ok", row
    physical = row["physical"]
    assert physical["scale_x"] == pytest.approx(1.0, abs=0.08)
    assert physical["scale_y"] == pytest.approx(1.0, abs=0.08)
    assert abs(physical["rotation_deg"]) < 3.0
    assert abs(physical["translate_x_mm"]) < 0.15 and abs(physical["translate_y_mm"]) < 0.15
    assert row["iou"] > 0.9
    assert row["regions"]["exclude"] == ["R"]
    assert row["regions"]["atlas_kept_fraction"] == pytest.approx(0.5, abs=0.05)
    assert row["regions"]["tissue_used_fraction"] > 0.95
    assert state.slices[0].transform["regions"] == {"include": [], "exclude": ["R"]}
    assert "kept regions" in kept["description"]


def test_include_restricts_and_says_what_it_cannot_measure(tmp_path: Path):
    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    row = _fit(box, ["s0.png"], "silhouette", include=["L"])["results"][0]
    assert row["status"] == "ok", row
    assert 0 < row["regions"]["outline_share"] < 1
    assert "Only the outline counts" in row["regions"]["note"]
    assert row["physical"]["scale_x"] == pytest.approx(1.0, abs=0.1)
    inside = _fit(box, ["s0.png"], "silhouette", include=["C"])["results"][0]
    assert inside["error"] == "REGIONS_INSIDE_OUTLINE"
    assert _fit(box, ["s0.png"], "silhouette", include=["nope"])["error"] == "UNKNOWN_REGIONS"
    assert _fit(box, ["s0.png"], "silhouette", include=["L"],
                exclude=["l"])["error"] == "BAD_ARGS"
    # Regions given, a damaged section is fitted; without them it is refused.
    state.slices[0].damaged = True
    assert _fit(box, ["s0.png"], "silhouette")["results"][0]["error"] == "DAMAGED"
    assert _fit(box, ["s0.png"], "silhouette", exclude=["R"])["status"] == "ok"


class WholeAtlas(HalvesAtlas):
    """The same ellipse as ONE region (``L``) across both hemispheres."""

    def __init__(self) -> None:
        super().__init__()
        self.annotation = np.where(self.annotation == RIGHT, LEFT, self.annotation)


def test_a_one_sided_exclusion_keeps_the_other_hemisphere(tmp_path: Path):
    """Excluding ``L:right`` drops only the right half of a region spanning both."""
    state, box = _box(tmp_path, WholeAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    row = _fit(box, ["s0.png"], "silhouette", exclude=["L:right"])["results"][0]
    assert row["status"] == "ok", row
    assert row["regions"]["atlas_kept_fraction"] == pytest.approx(0.5, abs=0.05)
    assert abs(row["physical"]["rotation_deg"]) < 3.0
    assert row["physical"]["scale_x"] == pytest.approx(1.0, abs=0.08)
    assert row["iou"] > 0.9
    assert state.slices[0].transform["regions"] == {"include": [], "exclude": ["L:right"]}
    # Excluding the side the tissue IS on leaves the wrong half to fit against.
    undo = next(t for t in box.tools if t.__name__ == "undo")
    undo()
    wrong = _fit(box, ["s0.png"], "silhouette", exclude=["L:left"])["results"][0]
    assert wrong["status"] == "ok" and wrong["regions"]["tissue_used_fraction"] < 0.2
    assert _fit(box, ["s0.png"], "silhouette", exclude=["L:middle"])["error"] == \
        "UNKNOWN_REGIONS"


def _tall_section() -> Image.Image:
    canvas = np.full((200, 160, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (80, 100), (60, 92), 0, 0, 360, (30, 30, 30), -1)
    return Image.fromarray(canvas, mode="RGB")


def test_a_picture_that_cannot_be_drawn_is_a_refusal_not_an_exception(tmp_path: Path):
    """A one-sided highlight on a placement turned past 45 degrees has no side."""
    state, box = _box(tmp_path, HalvesAtlas(), _tall_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    adjust = next(t for t in box.tools if t.__name__ == "adjust_transforms")
    assert adjust([{"id": "s0.png", "rotation_deg": 90, "scale_x": 1, "scale_y": 1,
                    "translate_x_mm": 0, "translate_y_mm": 0}])["status"] == "ok"
    before = dict(state.slices[0].transform or {})
    result = _fit(box, ["s0.png"], "elastix", view={"regions": ["L:left"]})
    assert result["error"] == "NOTHING_FITTED"
    assert result["results"][0]["error"] == "SIDES_AMBIGUOUS"
    assert state.slices[0].transform == before


def test_outlines_mode_draws_a_listed_atlas_image(tmp_path: Path):
    import io

    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.doors.tools.media import package_result

    _, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                  position_mm=1.0)
    view_placement = next(t for t in box.tools if t.__name__ == "view_placement")

    def mean(view: dict[str, Any]) -> float:
        part = package_result(
            view_placement([{"id": "s0.png"}], view=view))[TOOL_MEDIA_PARTS_KEY][0]
        return float(np.asarray(Image.open(io.BytesIO(part.inline_data.data)).convert("L")).mean())

    lines = mean({"mode": "outlines"})
    blended = mean({"mode": "outlines", "atlas_channels": ["ara", "borders"],
                    "atlas_opacity": 1.0})
    assert blended > lines + 20


def _flat_section() -> Image.Image:
    """A wide, flat ellipse: its included half turns ~90 degrees onto the atlas's."""
    canvas = np.full((200, 260, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (130, 100), (120, 30), 0, 0, 360, (30, 30, 30), -1)
    return Image.fromarray(canvas, mode="RGB")


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_large_turn_keeps_the_one_sided_highlight_the_fit_used(tmp_path: Path, side: str):
    """Sides are resolved once, by the fit, and the picture reuses them: a
    fit that turns the midline past 45 degrees still highlights its region
    (it used to fall back to no highlight at all)."""
    import io

    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.doors.tools.media import package_result

    def picture(view: dict[str, Any]) -> tuple[float, np.ndarray]:
        state, box = _box(tmp_path / f"{side}{len(view)}", WholeAtlas(), _flat_section(),
                          pixel_size_um=50.0, position_mm=1.0)
        result = _fit(box, ["s0.png"], "silhouette", include=[f"L:{side}"],
                      view={"mode": "outlines", **view})
        assert result["status"] == "ok", result
        part = package_result(result)[TOOL_MEDIA_PARTS_KEY][0]
        image = np.asarray(Image.open(io.BytesIO(part.inline_data.data)).convert("RGB"))
        return result["results"][0]["physical"]["rotation_deg"], image

    for folder in ("left0", "left1", "right0", "right1"):
        (tmp_path / folder).mkdir()
    turned, highlighted = picture({})
    _same, plain = picture({"regions": []})
    assert abs(turned) > 45
    # Full-strength yellow (the highlight) only where the region is drawn,
    # on the half of the canvas that side names.
    strong = (highlighted[..., 0] > 200) & (highlighted[..., 1] > 200) & (
        highlighted[..., 2] < 80)
    assert strong.sum() > 2 * ((plain[..., 0] > 200) & (plain[..., 1] > 200)
                               & (plain[..., 2] < 80)).sum()
    columns = np.nonzero(strong[60:].any(axis=0))[0]
    middle = highlighted.shape[1] / 2
    assert (columns.mean() < middle) == (side == "left")
