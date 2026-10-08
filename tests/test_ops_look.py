"""``ops.look``: look and zoom save every picture with a number, a caption and
a recipe; and ``ops.atlas.grep_atlas_view`` highlights named regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.layers import PictureNote
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState
from langslice.job.job import Job
from langslice.job.views import PICTURE_FILE, PictureRecord
from langslice.ops import atlas as ops_atlas
from langslice.ops import look as ops_look
from langslice.ops.refusal import Refused
from tests.deformable_synthetic import SyntheticAtlas
from tests.look_synthetic import MARKED, POSITIONS, SECTIONS, stack


def _record(job: Job, number: int) -> PictureRecord:
    found = job.views.lookup(number)
    assert found is not None, number
    return found


def _recipe(job: Job, number: int) -> dict[str, Any]:
    recipe = _record(job, number).recipe
    assert recipe is not None, number
    return recipe


def _section(job: Job, section_id: str) -> SliceState:
    found = job.state.by_id(section_id)
    assert found is not None
    return found


def _job(tmp_path: Path, *, persist: bool = True):
    ws0, state0 = stack(tmp_path, marked=True)
    atlas = SyntheticAtlas()
    spec = JobSpec(image_folder=ws0.image_folder, model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": 25.0})
    ws = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    job = Job.open(spec, ws, folder=ws.job_folder, results_path=ws.results_path)
    with job.writing():
        before = job.snapshot()
        for record in job.state.slices:
            source = state0.by_id(record.id)
            assert source is not None
            record.position_mm = source.position_mm
        job.commit(before)
    if not persist:  # a dry run's job: it writes nothing, its pictures in memory
        job = Job.load(spec, ws, folder=ws.job_folder, results_path=ws.results_path,
                       persist=False)
    return job, ws


def _move(job: Job, section_id: str, position: float) -> None:
    with job.writing():
        before = job.snapshot()
        record = job.state.by_id(section_id)
        assert record is not None
        record.position_mm = position
        job.commit(before)


def test_look_numbers_every_picture_and_indexes_its_recipe(tmp_path: Path):
    job, ws = _job(tmp_path)
    first = ops_look.look(job, ws, "section", sections=[SECTIONS[0], SECTIONS[1]])
    second = ops_look.look(job, ws, "overlay", sections=[SECTIONS[2]])
    assert [e["id"] for e in first.entries] == [1, 2]
    assert [e["id"] for e in second.entries] == [3]
    assert len(first.pictures) == 2 and not first.not_shown
    for entry in first.entries + second.entries:
        assert entry["caption"]
        found = job.views.lookup(entry["id"])
        assert found is not None and found.tool == "look" and found.caption == entry["caption"]
        assert found.recipe is not None and found.recipe["renderer"] == "look"
        json.dumps(found.recipe)
    assert _record(job, 2).sections == (SECTIONS[1],)
    assert _record(job, 3).mode == "overlay"
    # No arguments reads no state: every section, in stack order.
    every = ops_look.look(job, ws, "section")
    assert len(every.entries) == 4 == len(every.pictures)


def test_look_refusals(tmp_path: Path):
    job, ws = _job(tmp_path)
    cases = [
        (dict(mode="section", sections=["nope.png"]), "UNKNOWN_SLICE_IDS"),
        (dict(mode="sideways"), "UNKNOWN_MODE"),
        (dict(mode="atlas"), "NO_POSITIONS"),
        (dict(mode="section", warp="wavy"), "BAD_WARP"),
        (dict(mode="section", channels=["nope"]), "UNKNOWN_CHANNEL"),
        (dict(mode="atlas", positions_mm=[0.1], atlas_opacity=3), "BAD_ARGS"),
        (dict(mode="section", sections="s0.png"), "BAD_ARGS"),
    ]
    for kwargs, code in cases:
        with pytest.raises(Refused) as caught:
            ops_look.look(job, ws, **kwargs)  # type: ignore[arg-type]
        assert caught.value.code == code, (kwargs, caught.value.code)
    assert job.views.latest(exclude_tool=None) is None  # nothing saved


def test_more_than_four_pictures_are_named_not_shown(tmp_path: Path):
    job, ws = _job(tmp_path)
    positions = [0.05, 0.08, 0.11, 0.14, 0.17]
    looked = ops_look.look(job, ws, "atlas", positions_mm=positions)
    assert len(looked.pictures) == len(looked.entries) == ops_look.MAX_LOOK_PICTURES == 4
    (left_out,) = looked.not_shown
    assert left_out["positions_mm"] == [0.17] and left_out["mode"] == "atlas"
    assert "positions_mm=[0.17]" in left_out["how"] and left_out["caption"]
    assert [r.seq for r in job.views.records()] == [1, 2, 3, 4]  # the fifth was not saved


def test_zoom_defaults_to_the_latest_picture_that_is_not_a_zoom(tmp_path: Path):
    job, ws = _job(tmp_path)
    looked = ops_look.look(job, ws, "section", sections=[MARKED])
    (entry,) = looked.entries
    width, height = _recipe(job, entry["id"])["shown"]
    first = ops_look.zoom(job, ws, [0, 0, width / 2, height / 2])
    assert first.picture == entry["id"] and first.redrawn and not first.stale
    assert first.entries[0]["id"] == entry["id"] + 1
    saved = job.views.lookup(first.entries[0]["id"])
    assert saved is not None and saved.tool == "zoom" and saved.sections == (MARKED,)
    # The latest picture is now a zoom; the default still reads the look.
    second = ops_look.zoom(job, ws, [0, 0, width / 4, height / 4])
    assert second.picture == entry["id"]
    assert second.entries[0]["id"] == entry["id"] + 2
    with pytest.raises(Refused) as caught:
        ops_look.zoom(job, ws, [0, 0, 10, 10], picture=999)
    assert caught.value.code == "UNKNOWN_PICTURE"
    with pytest.raises(Refused) as caught:
        ops_look.zoom(job, ws, [5, 5, 5, 5], picture=entry["id"])
    assert caught.value.code == "EMPTY_BOX"


def test_zoom_lands_on_the_box_and_a_zoom_of_a_zoom_maps_back(tmp_path: Path):
    job, ws = _job(tmp_path)
    (entry,) = ops_look.look(job, ws, "section", sections=[MARKED]).entries
    recipe = _recipe(job, entry["id"])
    width, height = recipe["shown"]
    content = np.asarray(Image.open(job.layout.resolve(
        _record(job, entry["id"]).path) / PICTURE_FILE).convert("L"))[:height, :width]
    ys, xs = np.nonzero(content > 250)
    box = [xs.min() - 20, ys.min() - 20, xs.max() + 20, ys.max() + 20]
    zoomed = ops_look.zoom(job, ws, box, picture=entry["id"])
    assert zoomed.redrawn and zoomed.entries[0]["caption"]
    window = _recipe(job, zoomed.entries[0]["id"])["args"]["zoom"]
    assert window[0] == pytest.approx(box[0] / width, abs=1e-6)
    again = ops_look.zoom(job, ws, [0, 0, 10, 10], picture=zoomed.entries[0]["id"])
    inner = _recipe(job, again.entries[0]["id"])
    assert inner["base"] == recipe["base"]  # redrawn from the original, not enlarged
    assert inner["args"]["zoom"][0] == pytest.approx(window[0])
    assert inner["args"]["zoom"][2] < window[2]
    assert again.picture == zoomed.entries[0]["id"] and again.redrawn
    # The box was read on the picture: the white square fills the middle of
    # the zoom's content, framed by the slab's gray margin.
    zoom_width, zoom_height = _recipe(job, zoomed.entries[0]["id"])["shown"]
    shown = np.asarray(zoomed.pictures[0].convert("L"))[:zoom_height, :zoom_width]
    middle = shown[zoom_height * 2 // 5:zoom_height * 3 // 5,
                   zoom_width * 2 // 5:zoom_width * 3 // 5]
    assert middle.min() > 240
    assert shown[:, :zoom_width // 20].mean() < 200  # the margin left of the square


def test_a_changed_stack_is_redrawn_as_it_was_and_flagged_stale(tmp_path: Path):
    job, ws = _job(tmp_path)
    (entry,) = ops_look.look(job, ws, "overlay", sections=[SECTIONS[1]]).entries
    _move(job, SECTIONS[1], 0.22)
    zoomed = ops_look.zoom(job, ws, [0, 0, 50, 50])
    assert zoomed.stale and zoomed.redrawn and zoomed.picture == entry["id"]
    assert f"at {POSITIONS[1]:.2f} mm" in zoomed.entries[0]["caption"]
    assert "changed since" in zoomed.entries[0]["caption"]
    assert _section(job, SECTIONS[1]).position_mm == 0.22


def test_a_picture_without_a_recipe_is_cropped_and_a_crop_of_a_crop_maps_back(tmp_path: Path):
    job, ws = _job(tmp_path)
    source = Image.new("RGB", (400, 200), (0, 0, 0))
    source.paste((255, 255, 255), (300, 100, 400, 200))
    (name,) = job.views.save(tool="strip", pictures=[(source, PictureNote(mode="strip"))])
    number = int(name.split("_")[0])
    crop = ops_look.zoom(job, ws, [200, 0, 400, 200], picture=number)
    assert not crop.redrawn and crop.picture == number
    crop_id = crop.entries[0]["id"]
    held = job.views.lookup(crop_id)
    assert held is not None and held.recipe is not None
    assert held.recipe["renderer"] == "crop" and held.recipe["source"] == number
    width, height = held.recipe["shown"]
    again = ops_look.zoom(job, ws, [width / 2, height / 2, width, height], picture=crop_id)
    inner = _recipe(job, again.entries[0]["id"])
    assert inner["args"]["box"] == [300.0, 100.0, 400.0, 200.0] and inner["source"] == number
    assert np.asarray(again.pictures[0].convert("L"))[:10].min() > 240


def test_a_picture_that_is_not_saved_is_refused(tmp_path: Path):
    job, ws = _job(tmp_path)
    (name,) = job.views.save(tool="strip", pictures=[(Image.new("RGB", (40, 30)), None)])
    number = int(name.split("_")[0])
    job.views.flush()
    (job.layout.resolve(_record(job, number).path) / PICTURE_FILE).unlink()
    with pytest.raises(Refused) as caught:
        ops_look.zoom(job, ws, [0, 0, 20, 20], picture=number)
    assert caught.value.code == "PICTURE_NOT_SAVED"
    # A dry run keeps no pictures between processes: an unknown number is "not saved".
    (tmp_path / "dry").mkdir()
    dry, dry_ws = _job(tmp_path / "dry", persist=False)
    with pytest.raises(Refused) as caught:
        ops_look.zoom(dry, dry_ws, [0, 0, 20, 20], picture=7)
    assert caught.value.code == "PICTURE_NOT_SAVED"
    with pytest.raises(Refused) as caught:
        ops_look.zoom(dry, dry_ws, [0, 0, 20, 20])
    assert caught.value.code == "NO_PICTURE"


def test_a_dry_run_zooms_from_memory(tmp_path: Path):
    job, ws = _job(tmp_path, persist=False)
    (entry,) = ops_look.look(job, ws, "section", sections=[SECTIONS[0]]).entries
    zoomed = ops_look.zoom(job, ws, [0, 0, 60, 40])
    assert zoomed.picture == entry["id"] and zoomed.redrawn
    assert zoomed.entries[0]["id"] == entry["id"] + 1


def _yellow(picture: Image.Image) -> np.ndarray:
    data = np.asarray(picture.convert("RGB")).astype(int)
    return (data[..., 0] > 200) & (data[..., 1] > 200) & (data[..., 2] < 80)


def _centre_x(picture: Image.Image) -> float:
    _ys, xs = np.nonzero(_yellow(picture))
    assert xs.size > 20, "no highlighted border drawn"
    return float(xs.mean() / picture.width)


def test_grep_atlas_view_highlights_the_named_regions(tmp_path: Path):
    job, ws = _job(tmp_path)
    view = ops_atlas.grep_atlas_view(job, ws, ["STR"], [0.1, 0.3])
    assert view.regions == ["STR"] and len(view.pictures) == 2
    assert [e["id"] for e in view.entries] == [1, 2]
    assert view.clamped and view.clamped[0] == (0.3, pytest.approx(ws.position_range[1]))
    assert "regions highlighted: STR" in view.entries[0]["caption"]
    saved = job.views.lookup(1)
    assert saved is not None and saved.tool == "grep_atlas_view" and saved.mode == "atlas"
    assert saved.recipe is not None and saved.recipe["args"]["regions"] == ["STR"]
    striatum = _centre_x(view.pictures[0])
    thalamus = _centre_x(ops_atlas.grep_atlas_view(job, ws, ["TH"], [0.1]).pictures[0])
    assert striatum < 0.45 < 0.55 < thalamus  # STR sits left of TH in the synthetic plane
    # One side of a region: the highlight is the half on that side only.
    both = _yellow(ops_atlas.grep_atlas_view(job, ws, ["CTX"], [0.1]).pictures[0]).sum()
    left = _yellow(ops_atlas.grep_atlas_view(job, ws, ["CTX:left"], [0.1]).pictures[0]).sum()
    assert 0 < left < both
    assert not view.regions_not_in_plane


def test_grep_atlas_view_names_a_side_with_no_pixel_as_not_in_plane(tmp_path: Path):
    """TH lies wholly right of the midline: "TH:left" is drawn nowhere, and
    the reply says so, as it says for a region absent from the plane."""
    job, ws = _job(tmp_path)
    view = ops_atlas.grep_atlas_view(job, ws, ["TH:left", "TH:right", "STR"], [0.1])
    assert view.regions_not_in_plane == {"0.10": ["TH:left"]}


def test_grep_atlas_view_zooms_through_its_recipe_and_names_extra_pictures(tmp_path: Path):
    job, ws = _job(tmp_path)
    view = ops_atlas.grep_atlas_view(job, ws, ["TH", "STR"], [0.02, 0.05, 0.08, 0.11, 0.14])
    assert len(view.entries) == 4 and len(view.not_shown) == 1
    assert view.not_shown[0]["positions_mm"] == [0.14]
    shown = ops_look.zoom(job, ws, [0, 0, 120, 90], picture=view.entries[0]["id"])
    assert shown.redrawn and not shown.stale
    zoom_recipe = _recipe(job, shown.entries[0]["id"])
    assert zoom_recipe["renderer"] == "grep_atlas_view" and zoom_recipe["args"]["zoom"]
    assert "TH, STR" in shown.entries[0]["caption"]


def test_grep_atlas_view_refuses_unknown_regions(tmp_path: Path):
    job, ws = _job(tmp_path)
    for regions, code in ((["NOPE"], "UNKNOWN_REGIONS"), ([], "BAD_ARGS"), ("CTX", "BAD_ARGS")):
        with pytest.raises(Refused) as caught:
            ops_atlas.grep_atlas_view(job, ws, regions, [0.1])  # type: ignore[arg-type]
        assert caught.value.code == code
    with pytest.raises(Refused) as caught:
        ops_atlas.grep_atlas_view(job, ws, ["CTX"], [])
    assert caught.value.code == "BAD_ARGS"
    assert job.views.latest(exclude_tool=None) is None


def test_grep_atlas_view_has_no_context_borders(tmp_path: Path):
    job, ws = _job(tmp_path)
    view = ops_atlas.grep_atlas_view(job, ws, ["STR"], [0.1])
    image = np.asarray(view.pictures[0]).astype(int)
    # The template and caption are gray. Any colored pixel is a border,
    # including faint context lines; STR occupies only the left half.
    colored = image.max(axis=-1) - image.min(axis=-1) > 5
    assert colored[:, :image.shape[1] // 2].any()
    assert not colored[:, image.shape[1] // 2:].any()
