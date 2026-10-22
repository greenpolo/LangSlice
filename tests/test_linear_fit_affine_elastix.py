"""``elastix_affine``: a local intensity-affine refinement of the placement."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.affine import decompose_affine, normalized_affine
from langslice.core.canvas import canvas_geometry
from langslice.core.deformable import engines
from langslice.core.spec import JobSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from tests.deformable_synthetic import SECTION_SIZE, TH, SyntheticAtlas
from tests.linear_tool_helpers import tool_named as _tool

pytest.importorskip("itk", reason="elastix_affine needs itk-elastix")

ID = "s0.png"
UM = 25.0
POSITION = 0.1
WIDTH, HEIGHT = SECTION_SIZE


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _step(rotation_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    """A 3x3 map of section pixels (about the centre), section -> placed atlas."""
    angle = math.radians(rotation_deg)
    linear = scale * np.array([[math.cos(angle), math.sin(angle)],
                               [-math.sin(angle), math.cos(angle)]])
    centre = np.array([WIDTH / 2.0, HEIGHT / 2.0])
    step = np.eye(3)
    step[:2, :2] = linear
    step[:2, 2] = centre - linear @ centre + [tx, ty]
    return step


def _section(atlas: SyntheticAtlas, truth: np.ndarray, *,
             blank: tuple[int, ...] = ()) -> Image.Image:
    """The atlas drawn as a brightfield stain where the TRUE transform puts it.

    With the identity transform, the atlas plane lands on the section as
    ``placement`` (the canvas geometry ``elastix_affine`` uses); the section here
    shows, at pixel p, the atlas at ``truth @ p`` of that placement. Regions in
    *blank* are missing tissue (slide background).
    """
    geometry = canvas_geometry((WIDTH, HEIGHT), UM, atlas, POSITION, "coronal", 0.0, 0.0)
    s = geometry.atlas_scale
    (ax, ay), (sx, sy) = geometry.atlas_offset, geometry.section_offset
    placement = np.array([[s, 0.0, ax - sx], [0.0, s, ay - sy], [0.0, 0.0, 1.0]])
    native = np.linalg.inv(placement) @ truth
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    nx = (native[0, 0] * xx + native[0, 1] * yy + native[0, 2]).astype(np.float32)
    ny = (native[1, 0] * xx + native[1, 1] * yy + native[1, 2]).astype(np.float32)
    values = cv2.remap(atlas.template[0], nx, ny, cv2.INTER_LINEAR, borderValue=0)
    labels = cv2.remap(atlas.annotation[0].astype(np.float32), nx, ny, cv2.INTER_NEAREST,
                       borderValue=0)
    brightness = 236.0 - 0.85 * values
    brightness[(labels == 0) | np.isin(labels, blank)] = 236.0
    noise = np.random.default_rng(0).normal(0, 2.0, brightness.shape)
    gray = np.clip(brightness + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


def _setup(folder: Path, atlas: SyntheticAtlas, image: Image.Image):
    image.save(folder / ID)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    state.slices[0].position_mm = POSITION
    return state, ctx, build_tools(state, ctx, spec)


def _error_vs(truth: np.ndarray, params: list[float]) -> dict[str, float]:
    """How far a fitted transform is from *truth*: degrees, scale, pixels."""
    want = decompose_affine(normalized_affine(truth[:2], SECTION_SIZE), SECTION_SIZE)
    got = decompose_affine(params, SECTION_SIZE)
    fitted = np.vstack([np.array(params).reshape(2, 3), [0, 0, 1]])
    expected = np.vstack([np.array(normalized_affine(truth[:2], SECTION_SIZE)).reshape(2, 3),
                          [0, 0, 1]])
    corners = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=float)
    shift = np.abs((fitted - expected) @ corners)[:2] * np.array([[WIDTH], [HEIGHT]])
    return {"rotation": abs(got["rotation_deg"] - want["rotation_deg"]),
            "scale": max(abs(got["scale_x"] - want["scale_x"]),
                         abs(got["scale_y"] - want["scale_y"])),
            "corner_px": float(shift.max())}


def test_recovers_a_known_offset_and_records_it_like_any_fit(tmp_path: Path, atlas):
    truth = _step(4.0, 1.05, 6.0, -4.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth))
    result = _tool(box, "elastix_affine")([ID])
    assert result["status"] == "ok", result
    row = result["results"][0]
    assert row["status"] == "ok"
    error = _error_vs(truth, state.slices[0].transform["params"])
    assert error["rotation"] < 0.3 and error["scale"] < 0.01 and error["corner_px"] < 1.5, error
    # The same reply and record shape as a silhouette fit.
    assert set(row["transform"]) == {"rotation_deg", "scale_x", "scale_y", "shear",
                                    "translate_x_mm", "translate_y_mm", "pivot"}
    assert row["transform"]["translate_x_mm"] == pytest.approx(0.15, abs=0.01)
    assert row["transform"]["translate_y_mm"] == pytest.approx(-0.10, abs=0.01)
    assert row["mirrored"] is False
    assert row["calibration"] == {"section_um_per_px": UM, "source": "host"}
    assert "params" not in row  # the six raw numbers stay host-side
    # The section under its new transform, saved with a number.
    assert len(result["pictures"]) == 1 and len(result[TOOL_MEDIA_PARTS_KEY]) == 1
    stored = state.slices[0].transform
    assert stored["kind"] == "elastix" and len(stored["params"]) == 6
    assert "regions" not in stored and "regions" not in row
    # One undo step back to no transform; the checkpoint holds the fit.
    assert json.loads(Path(_ctx.checkpoint_path).read_text())["slices"][0]["transform"][
        "kind"] == "elastix"
    _tool(box, "undo")()
    assert state.slices[0].transform is None


def test_starts_from_the_current_placement(tmp_path: Path, atlas):
    """A refinement, never a search: from the true placement it stays put, and
    from a placement near it, it converges to it."""
    truth = _step(-3.0, 0.97, -5.0, 3.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth))
    record = state.slices[0]
    near = _step(-1.0, 1.0, -2.0, 1.0)
    record.transform = {"kind": "interactive", "params": normalized_affine(near[:2], SECTION_SIZE),
                        "calibration": {"section_um_per_px": UM, "source": "host"}}
    _tool(box, "elastix_affine")([ID], view=False)
    error = _error_vs(truth, record.transform["params"])
    assert error["rotation"] < 0.3 and error["corner_px"] < 1.5, error


def test_marked_damage_and_restrict_to_are_honoured(tmp_path: Path, atlas):
    """Thalamus missing from the section: marking it as damage (left out of the
    fit on its own) recovers the offset; restrict_to fits by its regions."""
    truth = _step(3.0, 1.04, 5.0, -3.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth, blank=(TH,)))
    assert _tool(box, "mark_damage")(ID, ["TH"])["status"] == "ok"
    excluded = _tool(box, "elastix_affine")([ID], view=False)
    row = excluded["results"][0]
    assert row["status"] == "ok", row
    assert row["regions"]["exclude"] == ["TH"] and row["regions"]["include"] == []
    assert 0.8 < row["regions"]["atlas_kept_fraction"] < 1.0
    assert state.slices[0].transform["regions"] == {"include": [], "exclude": ["TH"]}
    with_exclusion = _error_vs(truth, state.slices[0].transform["params"])
    assert with_exclusion["corner_px"] < 2.5, with_exclusion

    _tool(box, "undo")()
    _tool(box, "undo")()  # the mark too
    assert state.slices[0].damaged_regions == [] and state.slices[0].transform is None
    included = _tool(box, "elastix_affine")([ID], restrict_to=["STR"], view=False)
    row = included["results"][0]
    assert row["status"] == "ok", row
    assert row["regions"]["include"] == ["STR"]
    assert len(row["restrict_box"]) == 4  # where a picture zooms on the regions
    near_striatum = _error_vs(truth, state.slices[0].transform["params"])
    assert near_striatum["rotation"] < 1.5 and near_striatum["corner_px"] < 6.0, near_striatum

    _tool(box, "undo")()
    _tool(box, "mark_damage")(ID, ["TH:right"])
    one_side = _tool(box, "elastix_affine")([ID], view=False)
    assert one_side["results"][0]["status"] == "ok", one_side
    assert one_side["results"][0]["regions"]["exclude"] == ["TH:right"]


def test_an_elastix_failure_is_refused_cleanly(tmp_path: Path, atlas, monkeypatch):
    truth = _step(2.0, 1.0, 0.0, 0.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth))

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Elastix returned a non-finite or singular affine")

    monkeypatch.setattr(engines, "run_elastix_affine", broken)
    result = _tool(box, "elastix_affine")([ID])
    assert result["status"] == "error" and result["error"] == "FIT_FAILED"
    row = result["results"][0]
    assert row["error"] == "FIT_FAILED" and "non-finite" in row["message"]
    assert state.slices[0].transform is None
    assert TOOL_MEDIA_PARTS_KEY not in result


def _px(params: list[float]) -> np.ndarray:
    """Six normalized numbers as a 3x3 in section pixels."""
    from langslice.core.affine import denormalized_affine

    return np.vstack([denormalized_affine(params, SECTION_SIZE), [0, 0, 1]])


def test_a_manual_tweak_keeps_the_fits_shear(tmp_path: Path, atlas):
    """interactive_transform takes the shear elastix_affine reports: the fit's
    knobs given back store the fit's own map; leaving shear out keeps the fit's
    shear; an explicit 0 drops it."""
    centre = np.array([WIDTH / 2.0, HEIGHT / 2.0])
    slant = np.eye(3)
    slant[0, 1] = 0.06
    slant[:2, 2] = centre - slant[:2, :2] @ centre
    truth = _step(3.0, 1.02, 4.0, -2.0) @ slant
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth))
    row = _tool(box, "elastix_affine")([ID], view=False)["results"][0]
    fit = dict(state.slices[0].transform)
    fitted = list(fit["params"])
    knobs = {key: value for key, value in row["transform"].items() if key != "pivot"}
    assert abs(knobs["shear"]) > 0.03, knobs

    def tweak(**entry: Any) -> dict[str, Any]:
        result = _tool(box, "interactive_transform")([{"id": ID, **entry}], view=False)
        assert result["status"] == "ok", result
        return result["results"][0]["transform"]

    # The fit's knobs given back (shear included) redraw the fit's own map.
    assert tweak(**knobs)["shear"] == knobs["shear"]
    kept = _error_vs(_px(fitted), list(state.slices[0].transform["params"]))
    assert kept["corner_px"] < 0.2, kept
    assert tweak(**{**knobs, "shear": 0.0})["shear"] == 0.0
    dropped = _error_vs(_px(fitted), list(state.slices[0].transform["params"]))
    assert dropped["corner_px"] > 2.0, dropped

    # Only the five knobs changed: the fit's shear stays.
    state.slices[0].transform = fit
    five = {key: knobs[key] for key in ("rotation_deg", "scale_x", "scale_y",
                                        "translate_x_mm", "translate_y_mm")}
    reply = tweak(**{**five, "rotation_deg": five["rotation_deg"] + 0.5})
    assert reply["shear"] == knobs["shear"]
    assert decompose_affine(state.slices[0].transform["params"], SECTION_SIZE)["shear"] == \
        pytest.approx(knobs["shear"], abs=2e-3)
