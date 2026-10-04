"""`fit_affine`'s Elastix method: a local intensity-affine refinement of the placement."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import decompose_affine, normalized_affine
from langslice.deformable import engines
from langslice.linear import transform as tr
from langslice.linear.engine import build_context
from langslice.linear.job import ingest
from langslice.linear.render import canvas_geometry
from langslice.linear.spec import JobSpec
from langslice.linear.toolbox import build_tools
from tests.deformable_synthetic import SECTION_SIZE, TH, SyntheticAtlas

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
    ``placement`` (the canvas geometry ``fit_affine`` uses); the section here
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


def _setup(folder: Path, atlas: SyntheticAtlas, image: Image.Image, *, damaged: bool = False):
    image.save(folder / ID)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": UM})
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    state.slices[0].position_mm = POSITION
    state.slices[0].damaged = damaged
    return state, ctx, build_tools(state, ctx, spec)


def _tool(box: Any, name: str) -> Any:
    return next(tool for tool in box.tools if tool.__name__ == name)


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
    result = _tool(box, "fit_affine")([])  # elastix is the default method
    assert result["status"] == "ok", result
    row = result["results"][0]
    assert row["status"] == "ok"
    error = _error_vs(truth, state.slices[0].transform["params"])
    assert error["rotation"] < 0.3 and error["scale"] < 0.01 and error["corner_px"] < 1.5, error
    # The same reply and record shape as a silhouette fit.
    assert set(row["physical"]) == {"rotation_deg", "scale_x", "scale_y", "shear",
                                    "translate_x_mm", "translate_y_mm", "pivot"}
    assert row["physical"]["translate_x_mm"] == pytest.approx(0.15, abs=0.01)
    assert row["physical"]["translate_y_mm"] == pytest.approx(-0.10, abs=0.01)
    assert row["iou"] > 0.97 and row["mirrored"] is False
    assert row["calibration"] == {"section_um_per_px": UM, "source": "host"}
    assert row["image_indexes"] == [0] and len(result[TOOL_MEDIA_PARTS_KEY]) == 1
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
    _tool(box, "fit_affine")([ID])
    error = _error_vs(truth, record.transform["params"])
    assert error["rotation"] < 0.3 and error["corner_px"] < 1.5, error
    # From the true placement itself, the fit stays there.
    exact = normalized_affine(truth[:2], SECTION_SIZE)
    fit = tr.elastix_affine(state, _ctx, record, exact, record.transform["calibration"])
    assert fit.start_params == exact
    assert fit.start_iou > 0.97
    assert _error_vs(truth, fit.params)["corner_px"] < 1.0


def test_identical_inputs_give_identical_transforms(tmp_path: Path, atlas):
    truth = _step(2.0, 1.03, 3.0, 2.0)
    state, ctx, _box = _setup(tmp_path, atlas, _section(atlas, truth))
    record = state.slices[0]
    calibration = {"section_um_per_px": UM, "source": "host"}
    first = tr.elastix_affine(state, ctx, record, tr.IDENTITY_PARAMS, calibration)
    second = tr.elastix_affine(state, ctx, record, tr.IDENTITY_PARAMS, calibration)
    assert first.params == second.params  # bit for bit
    assert first.engine["native_parameters"]["random_seed"] == engines.RANDOM_SEED
    assert first.engine["native_parameters"]["threads"] == engines.FIT_THREADS
    assert first.engine["native_parameters"]["edge_channel"] is True


def test_exclude_and_include_are_honoured(tmp_path: Path, atlas):
    """Thalamus missing from the section: excluding it recovers the offset."""
    truth = _step(3.0, 1.04, 5.0, -3.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth, blank=(TH,)))
    excluded = _tool(box, "fit_affine")([ID], exclude=["TH"])
    row = excluded["results"][0]
    assert row["status"] == "ok", row
    assert row["regions"]["exclude"] == ["TH"] and row["regions"]["include"] == []
    assert 0.8 < row["regions"]["atlas_kept_fraction"] < 1.0
    assert state.slices[0].transform["regions"] == {"include": [], "exclude": ["TH"]}
    with_exclusion = _error_vs(truth, state.slices[0].transform["params"])
    assert with_exclusion["corner_px"] < 2.5, with_exclusion
    assert "kept regions" in excluded["description"]

    _tool(box, "undo")()
    included = _tool(box, "fit_affine")([ID], include=["STR"])
    row = included["results"][0]
    assert row["status"] == "ok", row
    assert row["regions"]["include"] == ["STR"]
    near_striatum = _error_vs(truth, state.slices[0].transform["params"])
    assert near_striatum["rotation"] < 1.5 and near_striatum["corner_px"] < 6.0, near_striatum

    _tool(box, "undo")()
    one_side = _tool(box, "fit_affine")([ID], exclude=["TH:right"])
    assert one_side["results"][0]["status"] == "ok", one_side
    assert one_side["results"][0]["regions"]["exclude"] == ["TH:right"]


def test_regions_reach_the_engine_as_the_deformable_fit_builds_them(
    tmp_path: Path, atlas, monkeypatch,
):
    """Excluded regions leave the atlas mask (blanked, with a margin); included
    ones limit both masks to their neighbourhood. Same masks as fit_deformable."""
    truth = _step(0.0, 1.0, 0.0, 0.0)
    _state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth))
    seen: list[Any] = []
    real = engines.run_elastix_affine

    def capture(inputs: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(inputs)
        return real(inputs, *args, **kwargs)

    monkeypatch.setattr(engines, "run_elastix_affine", capture)
    for regions in ({}, {"exclude": ["TH"]}, {"include": ["STR"]}):
        assert _tool(box, "fit_affine")([ID], **regions)["status"] == "ok"
        _tool(box, "undo")()
    whole, without, near = seen
    removed = whole.moving_mask & ~without.moving_mask
    assert removed.sum() > 0.05 * whole.moving_mask.sum()
    # The region itself is blanked; the rest of what left the mask is its margin.
    assert (without.moving[removed] == 0).mean() > 0.5
    assert np.array_equal(whole.fixed_mask, without.fixed_mask)  # tissue side by the map
    assert near.fixed_mask.sum() < 0.6 * whole.fixed_mask.sum()
    assert near.moving_mask.sum() < 0.6 * whole.moving_mask.sum()
    assert whole.fixed_edges is not None and whole.edge_weight == 1.0


def test_damaged_sections_need_regions(tmp_path: Path, atlas):
    truth = _step(2.0, 1.0, 0.0, 0.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth), damaged=True)
    assert _tool(box, "fit_affine")([ID])["results"][0]["error"] == "DAMAGED"
    fitted = _tool(box, "fit_affine")([ID], exclude=["HY"])
    assert fitted["results"][0]["status"] == "ok", fitted


def test_an_elastix_failure_is_refused_cleanly(tmp_path: Path, atlas, monkeypatch):
    truth = _step(2.0, 1.0, 0.0, 0.0)
    state, _ctx, box = _setup(tmp_path, atlas, _section(atlas, truth))

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Elastix returned a non-finite or singular affine")

    monkeypatch.setattr(engines, "run_elastix_affine", broken)
    result = _tool(box, "fit_affine")([ID])
    assert result["status"] == "error" and result["error"] == "NOTHING_FITTED"
    row = result["results"][0]
    assert row["error"] == "FIT_FAILED" and "non-finite" in row["message"]
    assert state.slices[0].transform is None
    assert TOOL_MEDIA_PARTS_KEY not in result


def test_a_section_without_tissue_is_refused_cleanly(tmp_path: Path, atlas):
    blank = Image.new("RGB", SECTION_SIZE, (236, 236, 236))
    state, _ctx, box = _setup(tmp_path, atlas, blank)
    result = _tool(box, "fit_affine")([ID])
    assert result["error"] == "NOTHING_FITTED"
    assert result["results"][0]["error"] == "FIT_FAILED"
    assert state.slices[0].transform is None
