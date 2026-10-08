"""The picture index: every saved picture carries a caption, a step and a
recipe, and can be looked up by its number (also in a dry run)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from langslice.core.layers import PictureNote, collecting, note
from langslice.job.layout import JobLayout
from langslice.job.views import DiscardedViews, ViewStore, captured

RECIPE = {"renderer": "section_picture", "args": {"id": "s0.png", "long_edge": 512}}


def _picture(shade: int = 90) -> Image.Image:
    return Image.new("RGB", (16, 12), (shade, shade, shade))


def _store(tmp_path: Path, cls: type[ViewStore] = ViewStore) -> ViewStore:
    layout = JobLayout(tmp_path / "langslice")
    layout.folder.mkdir(parents=True)
    return cls(layout)


def test_view_json_and_the_index_carry_caption_step_and_recipe(tmp_path: Path):
    store = _store(tmp_path)
    store.step_source = lambda: 4
    with captured() as saved:
        store.save(tool="view", pictures=[
            (_picture(), PictureNote(sections=("s0.png",), mode="section", recipe=RECIPE,
                                     caption="s0 as drawn")),
            (_picture(), None)])
    store.flush()
    assert [picture.seq for picture in saved] == [1, 2]
    line = json.loads((tmp_path / "langslice" / "views.jsonl").read_text().splitlines()[0])
    assert (line["caption"], line["step"], line["recipe"]) == ("s0 as drawn", 4, RECIPE)
    record = json.loads((tmp_path / "langslice" / line["path"] / "view.json").read_text())
    assert (record["caption"], record["step"], record["recipe"]) == ("s0 as drawn", 4, RECIPE)
    bare = json.loads((tmp_path / "langslice" / "views.jsonl").read_text().splitlines()[1])
    assert "caption" not in bare and "recipe" not in bare and bare["step"] == 4


def test_note_carries_recipe_and_caption_and_old_callers_still_work():
    image = _picture()
    plain = _picture()
    with collecting() as notes:
        note(image, sections=["s1.png"], mode="atlas", recipe=RECIPE, caption="c")
        note(plain, mode="section")
    held = notes[0][1]
    assert held.recipe == RECIPE and held.caption == "c" and held.sections == ("s1.png",)
    assert notes[1][1].recipe is None and notes[1][1].caption is None


def test_lookup_and_latest_skip_zoom_and_survive_a_new_store(tmp_path: Path):
    store = _store(tmp_path)
    store.save(tool="view", pictures=[(_picture(), PictureNote(recipe=RECIPE))])
    store.save(tool="zoom", pictures=[(_picture(), None)])
    found = store.lookup(1)
    assert found is not None and found.tool == "view" and found.recipe == RECIPE
    assert store.lookup(99) is None
    latest = store.latest()
    assert latest is not None and latest.seq == 1
    newest = store.latest(exclude_tool=None)
    assert newest is not None and newest.seq == 2
    reopened = ViewStore(store.layout)
    again = reopened.lookup(2)
    assert again is not None and again.tool == "zoom"


def test_a_dry_run_indexes_pictures_in_memory(tmp_path: Path):
    store = _store(tmp_path, DiscardedViews)
    store.step_source = lambda: 7
    assert store.save(tool="view", pictures=[
        (_picture(), PictureNote(sections=("s0.png",), recipe=RECIPE, caption="x"))]) == [
            "000001_view"]  # named as a saved picture would be, written nowhere
    store.save(tool="zoom", pictures=[(_picture(), None)])
    found = store.lookup(1)
    assert found is not None and found.recipe == RECIPE and found.step == 7
    assert found.caption == "x" and found.sections == ("s0.png",) and found.image is not None
    latest = store.latest()
    assert latest is not None and latest.seq == 1
    assert not (tmp_path / "langslice" / "views.jsonl").exists()


def test_a_dry_run_store_keeps_only_the_newest_images(tmp_path: Path):
    store = _store(tmp_path, DiscardedViews)
    for _ in range(DiscardedViews.KEPT_PICTURES + 3):
        store.save(tool="view", pictures=[(_picture(), None)])
    first, last = store.lookup(1), store.lookup(DiscardedViews.KEPT_PICTURES + 3)
    assert first is not None and first.image is None
    assert last is not None and last.image is not None
