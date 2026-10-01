"""Multi-channel host snapshots: one blend for the preview and for the agent."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.api.abba_worker import run_linear
from langslice.api.service import run_stdio
from langslice.image_prep import (
    adaptive_preprocess,
    host_preprocess,
    host_preprocess_file,
    normalize_image,
    read_pages,
)


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
def test_auto_on_one_page_is_todays_automatic_path(tmp_path: Path, page: np.ndarray):
    path = tmp_path / "one.tif"
    tifffile.imwrite(path, page, photometric="rgb" if page.ndim == 3 else "minisblack")
    with Image.open(path) as handle:
        today = adaptive_preprocess(normalize_image(handle.copy()))
    for settings in (None, {"mode": "auto"}):
        result = host_preprocess_file(path, settings)
        assert np.array_equal(np.asarray(result), np.asarray(today))


def test_auto_weights_the_channel_that_lights_the_tissue():
    broad, sparse = _two_channels()
    blended = np.asarray(host_preprocess([broad, sparse]))[..., 0].astype(float)
    only_broad = np.asarray(
        host_preprocess([broad, sparse], {"mode": "custom", "channel_weights": [1, 0]})
    )[..., 0].astype(float)
    only_sparse = np.asarray(
        host_preprocess([broad, sparse], {"mode": "custom", "channel_weights": [0, 1]})
    )[..., 0].astype(float)
    assert np.corrcoef(blended.ravel(), only_broad.ravel())[0, 1] > 0.95
    assert np.corrcoef(blended.ravel(), only_sparse.ravel())[0, 1] < 0.5


def test_custom_weights_clahe_and_strength_change_the_picture():
    broad, sparse = _two_channels()
    base = {"mode": "custom", "channel_weights": [1, 1]}
    even = np.asarray(host_preprocess([broad, sparse], base))
    plain = np.asarray(host_preprocess([broad, sparse], {**base, "clahe": False}))
    strong = np.asarray(host_preprocess([broad, sparse], {**base, "clahe_strength": "high"}))
    assert not np.array_equal(even, plain)
    assert not np.array_equal(even, strong)
    # Weights are relative: doubling both changes nothing.
    doubled = np.asarray(host_preprocess([broad, sparse], {**base, "channel_weights": [2, 2]}))
    assert np.array_equal(even, doubled)
    assert even.shape == (*broad.shape, 3)


@pytest.mark.parametrize("settings, message", [
    ({"mode": "custom", "channel_weights": [1]}, "Expected 2"),
    ({"mode": "custom", "channel_weights": [1, -1]}, "non-negative"),
    ({"mode": "custom", "channel_weights": [0, 0]}, "above zero"),
    ({"mode": "custom", "clahe_strength": "max"}, "strength"),
    ({"mode": "fancy"}, "mode"),
])
def test_bad_settings_are_refused(settings: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        host_preprocess(list(_two_channels()), settings)


def test_pages_must_share_one_size():
    with pytest.raises(ValueError, match="one size"):
        host_preprocess([_tissue(), _tissue((60, 80))])


def test_a_multi_page_tiff_reads_one_array_per_page(tmp_path: Path):
    path = tmp_path / "two.tif"
    tifffile.imwrite(path, np.stack(_two_channels()))
    pages = read_pages(path)
    assert len(pages) == 2
    assert np.array_equal(pages[1], _two_channels()[1])


# --- preprocess.preview ----------------------------------------------------


def _request(*lines: dict[str, Any]) -> list[dict[str, Any]]:
    output = io.StringIO()
    run_stdio(input_stream=io.StringIO("\n".join(json.dumps(x) for x in lines) + "\n"),
              output_stream=output)
    return [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]


def test_preview_writes_exactly_the_blend(tmp_path: Path):
    path = tmp_path / "snap.tif"
    tifffile.imwrite(path, np.stack(_two_channels()))
    settings = {"mode": "custom", "clahe": True, "clahe_strength": "low",
                "channel_weights": [0.7, 0.3]}
    out = tmp_path / "preview" / "snap.png"
    [message] = _request({"id": "p", "method": "preprocess.preview", "params": {
        "image_path": str(path), "preprocessing": settings, "output_path": str(out)}})
    assert message["result"] == {"output_path": str(out), "width": 160, "height": 120}
    written = np.asarray(Image.open(out))
    expected = np.asarray(host_preprocess(read_pages(path), settings).convert("L"))
    assert np.array_equal(written, expected)


def test_preview_refuses_bad_settings_as_validation_errors(tmp_path: Path):
    [message] = _request({"id": "p", "method": "preprocess.preview", "params": {
        "image_path": str(tmp_path / "x.tif"), "output_path": str(tmp_path / "x.png"),
        "preprocessing": {"mode": "custom", "channel_weights": [-1]}}})
    assert message["error"]["code"] == "validation_error"


# --- linear.run staging and host updates -----------------------------------


@pytest.fixture
def snapshots(tmp_path: Path):
    tifffile.imwrite(tmp_path / "a.tif", np.stack(_two_channels()))
    tifffile.imwrite(tmp_path / "b.tif", np.stack(_two_channels()[::-1]))
    return {"image_folder": str(tmp_path), "pixel_size_um": 25,
            "positions_mm": {"a.tif": 4.0, "b.tif": 4.2},
            "spec": {"tasks": ["position", "transform"]}}


def _fake_engine(monkeypatch, steps, seen: dict[str, Any]):
    """Replace the engine: record the spec, then write each step as a checkpoint."""
    from langslice.linear import engine

    async def run(spec, *, on_write, on_event, emit):
        seen["spec"] = spec
        rows = [{"id": name, "position_mm": value, "index_corrected": index,
                 "flip": False, "rotation_deg": 0,
                 "transform": (spec.inputs.get("transforms") or {}).get(name)}
                for index, (name, value) in enumerate(sorted(spec.inputs["positions"].items()))]
        value = {"slices": rows}
        state = SimpleNamespace(to_dict=lambda: copy.deepcopy(value))
        on_write(state)
        for step in steps:
            step(rows)
            on_write(state)
        return state

    monkeypatch.setattr(engine, "run", run)


def test_multi_page_snapshots_show_the_agent_the_preview_blend(snapshots, monkeypatch):
    """The host blend is the DEFAULT appearance; the pages stay raw channels."""
    from langslice.linear.engine import build_context
    from tests.fakes import SlabAtlas

    seen: dict[str, Any] = {}
    _fake_engine(monkeypatch, [], seen)
    settings = {"mode": "custom", "channel_weights": [1, 3]}
    run_linear({**snapshots, "preprocessing": settings}, lambda event: None)
    spec = seen["spec"]
    folder = Path(snapshots["image_folder"])
    # Nothing is staged: the run reads the snapshots themselves.
    assert Path(spec.image_folder) == folder
    assert not (folder / "agent_view").exists()
    assert spec.host_preprocessing["channel_weights"] == [1, 3]
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: SlabAtlas())
    shown = np.asarray(ctx.working_source("a.tif")[0].convert("L"))
    preview = np.asarray(host_preprocess(read_pages(folder / "a.tif"), settings).convert("L"))
    assert np.array_equal(shown, preview)
    names, planes = ctx.section_channels("a.tif")
    assert names == ("ch1", "ch2")
    broad, sparse = _two_channels()
    assert np.array_equal(planes[0], broad) and np.array_equal(planes[1], sparse)


def test_host_channel_names_name_the_raw_channels(snapshots, monkeypatch):
    from langslice.linear.engine import build_context
    from tests.fakes import SlabAtlas

    seen: dict[str, Any] = {}
    _fake_engine(monkeypatch, [], seen)
    run_linear({**snapshots, "channel_names": ["DAPI", "tdTomato"]}, lambda event: None)
    ctx = build_context(seen["spec"], emit=lambda _m: None, atlas_loader=lambda _n: SlabAtlas())
    assert ctx.section_channels("b.tif")[0] == ("DAPI", "tdTomato")


def test_single_page_snapshots_without_settings_keep_todays_path(tmp_path, monkeypatch):
    Image.fromarray(_tissue()).save(tmp_path / "a.tif")
    seen: dict[str, Any] = {}
    _fake_engine(monkeypatch, [], seen)
    run_linear({"image_folder": str(tmp_path), "pixel_size_um": 25,
                "positions_mm": {"a.tif": 4.0}, "spec": {"tasks": ["position"]}},
               lambda event: None)
    assert Path(seen["spec"].image_folder) == tmp_path
    assert seen["spec"].preprocess == "auto"
    assert seen["spec"].host_preprocessing is None


def test_bad_weights_for_the_pages_are_refused_before_the_engine(snapshots, monkeypatch):
    seen: dict[str, Any] = {}
    _fake_engine(monkeypatch, [], seen)
    with pytest.raises(ValueError, match="Expected 2"):
        run_linear({**snapshots, "preprocessing": {"mode": "custom", "channel_weights": [1]}},
                   lambda event: None)
    assert "spec" not in seen


def test_checkpoints_carry_updates_since_start_and_the_result_the_final_ones(
    snapshots, monkeypatch,
):
    def move(value):
        def step(rows):
            rows[0]["position_mm"] = value
        return step

    def turn_locked(rows):
        rows[1]["transform"] = {"params": [1, 0, .1, 0, 1, 0]}

    seen: dict[str, Any] = {}
    _fake_engine(monkeypatch, [move(4.1), move(4.3), turn_locked], seen)
    events: list[dict[str, Any]] = []
    result = run_linear(
        {**snapshots, "locked": ["b.tif"], "damaged": {"a.tif": "torn"}}, events.append,
    )
    assert seen["spec"].inputs["locked"] == ["b.tif"]
    assert seen["spec"].inputs["damaged"] == {"a.tif": "torn"}
    checkpoints = [event for event in events if event["kind"] == "checkpoint"]
    assert checkpoints[0]["initial"] and checkpoints[0]["updates_since_start"] == []
    assert checkpoints[2]["host_updates"] == [{"id": "a.tif", "position_mm": 4.3}]
    assert checkpoints[2]["updates_since_start"] == [{"id": "a.tif", "position_mm": 4.3}]
    # A locked section's transform is never sent back to the host.
    assert checkpoints[3]["host_updates"] == []
    assert result["final_updates"] == [{"id": "a.tif", "position_mm": 4.3}]
    assert checkpoints[-1]["updates_since_start"] == result["final_updates"]


def test_reorder_with_registered_slices_is_no_longer_refused(snapshots, monkeypatch):
    seen: dict[str, Any] = {}
    _fake_engine(monkeypatch, [], seen)
    run_linear({**snapshots, "registered_slices": ["a.tif"],
                "spec": {"tasks": ["reorder", "position"]}}, lambda event: None)
    assert seen["spec"].tasks == ["reorder", "position"]
