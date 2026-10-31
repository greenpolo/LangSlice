"""Multi-channel host snapshots: the automatic blend the agent sees first."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.core.sections.reading import (
    adaptive_preprocess,
    custom_appearance,
    host_preprocess,
    normalize_image,
    read_pages,
)
from langslice.doors.abba.contract import AbbaPrepareRequest
from langslice.doors.abba.prepare import prepare_job
from langslice.doors.jobs import read_spec


def _tissue(shape=(120, 160)) -> np.ndarray:
    """A bright ellipse of tissue with some internal structure on black."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    body = ((yy - 60) / 45.0) ** 2 + ((xx - 80) / 65.0) ** 2 < 1
    texture = (np.sin(xx / 5.0) * 30 + 90).astype(np.uint8)
    return np.where(body, texture, 0).astype(np.uint8)


def _two_channels() -> tuple[np.ndarray, np.ndarray]:
    broad = _tissue()  # lights the whole tissue, like a nuclear stain
    sparse = np.zeros_like(broad)
    sparse[50:60, 70:80] = 250  # a small bright tracer patch
    return broad, sparse


# --- the function --------------------------------------------------------


@pytest.mark.parametrize("page", [
    _tissue(),
    (_tissue().astype(np.uint16) * 200),
    np.stack([_tissue(), _tissue() // 2, _tissue() // 3], axis=-1),
])
def test_auto_on_one_page_is_the_automatic_path(tmp_path: Path, page: np.ndarray):
    path = tmp_path / "one.tif"
    tifffile.imwrite(path, page, photometric="rgb" if page.ndim == 3 else "minisblack")
    with Image.open(path) as handle:
        automatic = adaptive_preprocess(normalize_image(handle.copy()))
    for settings in (None, {"mode": "auto"}):
        result = host_preprocess(read_pages(path), settings)
        assert np.array_equal(np.asarray(result), np.asarray(automatic))


def test_auto_weights_the_channel_that_lights_the_tissue():
    broad, sparse = _two_channels()
    blended = np.asarray(host_preprocess([broad, sparse]))[..., 0].astype(float)
    only_broad = np.asarray(
        custom_appearance([broad, sparse], channel_weights=[1, 0]))[..., 0].astype(float)
    only_sparse = np.asarray(
        custom_appearance([broad, sparse], channel_weights=[0, 1]))[..., 0].astype(float)
    assert np.corrcoef(blended.ravel(), only_broad.ravel())[0, 1] > 0.95
    assert np.corrcoef(blended.ravel(), only_sparse.ravel())[0, 1] < 0.5


@pytest.mark.parametrize("settings", [
    {"mode": "custom", "channel_weights": [1, 1]},
    {"mode": "fancy"},
])
def test_a_blend_other_than_the_automatic_one_is_refused(settings: dict[str, Any]):
    with pytest.raises(ValueError, match="channel blend must be"):
        host_preprocess(list(_two_channels()), settings)


def test_an_automatic_blend_saved_with_the_dialogs_keys_is_the_automatic_one():
    """Jobs an ABBA dialog made saved the automatic mode with its other keys."""
    pages = list(_two_channels())
    saved = {"mode": "auto", "clahe": True, "clahe_strength": "medium",
             "channel_weights": None}
    assert np.array_equal(np.asarray(host_preprocess(pages, saved)),
                          np.asarray(host_preprocess(pages)))


def test_pages_must_share_one_size():
    with pytest.raises(ValueError, match="one size"):
        host_preprocess([_tissue(), _tissue((60, 80))])


def test_a_multi_page_tiff_reads_one_array_per_page(tmp_path: Path):
    path = tmp_path / "two.tif"
    tifffile.imwrite(path, np.stack(_two_channels()))
    pages = read_pages(path)
    assert len(pages) == 2
    assert np.array_equal(pages[1], _two_channels()[1])


# --- a prepared ABBA job's snapshots ----------------------------------------


def _prepare(folder: Path, names: list[str], **extra: Any) -> Path:
    request = AbbaPrepareRequest.model_validate({
        "image_folder": str(folder), "pixel_size_um": 25,
        "positions_mm": {name: 4.0 + 0.1 * index for index, name in enumerate(names)},
        "angles_deg": {"pitch_deg": 0.0, "yaw_deg": 0.0},
        "abba": {"project_key": "k", "atlas": "Allen Mouse V3p1", "sections": {}},
        **extra})
    return prepare_job(request)


def test_multi_page_snapshots_open_on_the_automatic_blend(tmp_path: Path):
    """The blend is the DEFAULT appearance; the pages stay raw channels."""
    from langslice.doors.jobs import context
    from tests.fakes import SlabAtlas

    tifffile.imwrite(tmp_path / "a.tif", np.stack(_two_channels()))
    tifffile.imwrite(tmp_path / "b.tif", np.stack(_two_channels()[::-1]))
    job_folder = _prepare(tmp_path, ["a.tif", "b.tif"], channel_names=["DAPI", "tdTomato"])
    spec = read_spec(job_folder)
    assert spec.host_preprocessing == {"mode": "auto"}
    ctx = context(spec, job_folder, atlas_loader=lambda _n: SlabAtlas())
    shown = np.asarray(ctx.working_source("a.tif")[0].convert("L"))
    blend = np.asarray(host_preprocess(read_pages(tmp_path / "a.tif")).convert("L"))
    assert np.array_equal(shown, blend)
    names, planes = ctx.section_channels("a.tif")
    assert names == ("DAPI", "tdTomato")
    broad, sparse = _two_channels()
    assert np.array_equal(planes[0], broad) and np.array_equal(planes[1], sparse)


def test_single_page_snapshots_keep_the_automatic_path(tmp_path: Path):
    Image.fromarray(_tissue()).save(tmp_path / "a.tif")
    spec = read_spec(_prepare(tmp_path, ["a.tif"]))
    assert spec.preprocess == "auto"
    assert spec.host_preprocessing is None
