"""Round-4 trial fixes: the region at a point of a saved picture, a zoom that a
damage mark does not make stale, and the CLI's box and grep_atlas rows."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from langslice.ops import atlas, damage, look
from langslice.ops.refusal import Refused
from tests.test_agent_cli import cli, images, init, stack  # noqa: F401
from tests.test_landmarks import setup


def _overlay(job, ws, record):
    shown = look.look(job, ws, "overlay", sections=[record.id])
    job.views.flush()
    return shown.entries[0]["id"]


def _labels(job, seq):
    folder = job.layout.resolve(job.views.lookup(seq).path)
    return tifffile.imread(folder / "labels.tif")


def _inside_point(job, seq):
    """Pixel centre (x, y) with a region drawn, and the id drawn there."""
    labels = _labels(job, seq)
    rows, cols = np.nonzero(labels)
    k = len(rows) // 2
    return float(cols[k]), float(rows[k]), int(labels[rows[k], cols[k]])


def test_the_region_at_a_point_is_what_the_overlay_drew(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        seq = _overlay(job, ws, record)
        x, y, region_id = _inside_point(job, seq)
        rows = atlas.region_at_points(job, ws, [{"picture": seq, "x": x, "y": y}])
        entry = ws.atlas.structures[region_id]
        assert rows[0]["region"] == {"acronym": entry["acronym"], "name": entry["name"],
                                     "id": region_id}
        assert isinstance(rows[0]["within"], list) and len(rows[0]["within"]) <= atlas.MAX_CHAIN
        outside = atlas.region_at_points(job, ws, [{"picture": seq, "x": 0.0, "y": 0.0}])
        assert outside[0]["region"] is None and outside[0]["note"]
        reply = atlas.grep_atlas(job, ws, points=[{"picture": seq, "x": x, "y": y}])
        assert reply["points"][0]["region"]["id"] == region_id and "rows" not in reply
    finally:
        job.close()


def test_a_zoom_grid_point_is_the_same_point_as_on_the_source_picture(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        seq = _overlay(job, ws, record)
        x, y, _ = _inside_point(job, seq)
        box = [x - 40, y - 30, x + 40, y + 30]
        zoomed = look.zoom(job, ws, box=box, picture=seq)
        job.views.flush()
        zoom_seq = zoomed.entries[0]["id"]
        on_source = atlas.region_at_points(job, ws, [{"picture": seq, "x": x, "y": y}])
        on_zoom = atlas.region_at_points(job, ws, [{"picture": zoom_seq, "x": x, "y": y}])
        assert on_zoom[0]["region"] == on_source[0]["region"]
        assert on_zoom[0]["region"] is not None
    finally:
        job.close()


def test_a_section_only_picture_is_refused_with_the_fix(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        shown = look.look(job, ws, "section", sections=[record.id])
        job.views.flush()
        seq = shown.entries[0]["id"]
        with pytest.raises(Refused) as caught:
            atlas.region_at_points(job, ws, [{"picture": seq, "x": 50, "y": 50}])
        assert caught.value.code == "NO_ATLAS_IN_PICTURE"
        assert "overlay" in str(caught.value)
        with pytest.raises(Refused) as caught:
            atlas.region_at_points(job, ws, [{"picture": 999, "x": 1, "y": 1}])
        assert caught.value.code == "UNKNOWN_PICTURE"
        with pytest.raises(Refused, match="At most"):
            atlas.region_at_points(
                job, ws, [{"picture": seq, "x": 1, "y": 1}] * (atlas.MAX_POINTS + 1))
        with pytest.raises(Refused, match="Empty query"):
            atlas.grep_atlas(job, ws)
    finally:
        job.close()


def test_an_atlas_picture_drawn_for_a_section_gives_the_region(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        shown = look.look(job, ws, "atlas", sections=[record.id])
        job.views.flush()
        seq = shown.entries[0]["id"]
        saved = job.views.lookup(seq)
        assert saved is not None and saved.recipe is not None
        panel = saved.recipe["endpoint_panels"][0]
        x0, y0, x1, y1 = panel["bounds"]
        found = [r for r in atlas.region_at_points(
            job, ws, [{"picture": seq, "x": x0 + (x1 - x0) * f, "y": y0 + (y1 - y0) * g}
                      for f in (0.3, 0.5, 0.7) for g in (0.3, 0.5, 0.7)])
            if r["region"] is not None]
        assert found and all("acronym" in r["region"] for r in found)
    finally:
        job.close()


def test_marking_damage_does_not_make_a_zoom_stale(tmp_path):
    job, ws, record = setup(tmp_path)
    try:
        seq = _overlay(job, ws, record)
        acronym = next(iter(ws.atlas.structures.values()))["acronym"]
        # any region of the synthetic atlas
        regions = [str(v["acronym"]) for v in ws.atlas.structures.values()][1:2] or [acronym]
        damage.mark_damage(job, ws, record.id, regions)
        zoomed = look.zoom(job, ws, box=[10, 10, 100, 100], picture=seq)
        assert zoomed.stale is False
        moved = job.state.by_id(record.id)
        assert moved is not None and moved.position_mm is not None
        moved.position_mm += 0.05  # a placement change does
        zoomed = look.zoom(job, ws, box=[10, 10, 100, 100], picture=seq)
        assert zoomed.stale is True
    finally:
        job.close()


def test_the_cli_accepts_a_comma_box_and_returns_grep_atlas_rows(capsys, images):  # noqa: F811
    folder = init(capsys, images)
    code, shown = cli(capsys, str(folder), "look", "--mode", "section", "--sections", "s0.png")
    assert code == 0, shown
    number = shown["result"]["pictures"][0]["id"]
    for box in ("10,10,120,100", "[10,10,120,100]"):
        code, zoomed = cli(capsys, str(folder), "zoom", "--box", box, "--picture", str(number))
        assert code == 0, zoomed
    code, found = cli(capsys, str(folder), "grep_atlas", "--query", "root")
    assert code == 0 and isinstance(found["result"]["rows"], list) and found["result"]["rows"]
