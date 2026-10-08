"""look's four modes (core/look.py) on the synthetic atlas.

One picture per section (section, overlay) or per position (atlas); every
picture carries a JSON recipe and an index caption, noted for the picture
index without losing a placement picture's frame; defaults never read the
job's state; the snapshot puts back exactly what a picture was drawn from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langslice.core.channels import ChannelProperties, set_properties
from langslice.core.layers import collecting, note_for
from langslice.core.look import (
    LookError,
    LookRequest,
    changed_since,
    look,
    restored,
    snapshot,
)
from tests.look_synthetic import SECTIONS, record, stack


def _recipe_ok(picture, mode: str) -> dict:
    recipe = json.loads(json.dumps(picture.recipe))
    assert recipe == picture.recipe  # JSON as it is
    assert recipe["renderer"] == "look" and recipe["args"]["mode"] == mode
    assert recipe["args"]["zoom"] == []
    assert recipe["shown"] == recipe["base"]
    assert recipe["um_per_px"] == pytest.approx(picture.um_per_px, rel=1e-3)
    return recipe


def test_section_mode_one_picture_per_section(tmp_path: Path):
    ws, state = stack(tmp_path)
    with collecting() as notes:
        pictures = look(ws, state, LookRequest("section"))
    assert [p.sections for p in pictures] == [(name,) for name in SECTIONS]
    for picture in pictures:
        recipe = _recipe_ok(picture, "section")
        assert recipe["args"]["sections"] == list(picture.sections)
        assert list(recipe["state"]["sections"]) == list(picture.sections)
        assert "view_angles" not in recipe["state"]
        assert "25.0 um/px" in picture.caption and "raw gray" in picture.caption
        held = note_for(picture.image, notes)
        assert held is not None and held.recipe == picture.recipe
        assert held.caption == picture.caption and held.mode == "section"


def test_section_channels_preprocessed_and_errors(tmp_path: Path):
    ws, state = stack(tmp_path)
    one = ("s0.png",)
    picture = look(ws, state, LookRequest("section", sections=one, channels=("preprocessed",)))[0]
    assert "preprocessed channel (default appearance)" in picture.caption
    for channels, code in ((("DAPI",), "UNKNOWN_CHANNEL"),
                           (("gray", "preprocessed"), "MIXED_CHANNELS")):
        with pytest.raises(LookError) as caught:
            look(ws, state, LookRequest("section", sections=one, channels=channels))
        assert caught.value.code == code
    with pytest.raises(LookError) as caught:
        look(ws, state, LookRequest("section", sections=("nope.png",)))
    assert caught.value.code == "UNKNOWN_SECTION"
    with pytest.raises(LookError) as caught:
        look(ws, state, LookRequest("sideways"))
    assert caught.value.code == "UNKNOWN_MODE"


def test_display_properties_restated_in_the_caption(tmp_path: Path):
    ws, state = stack(tmp_path)
    set_properties(state, "gray", ChannelProperties(contrast_limits=(20.0, 200.0), gamma=1.5))
    picture = look(ws, state, LookRequest("section", sections=("s0.png",)))[0]
    assert "gray 20-200 gamma 1.5" in picture.caption
    assert "percentile" not in picture.caption
    assert picture.recipe["state"]["appearance"]["channels"]["gray"]["gamma"] == 1.5


def test_atlas_mode_needs_positions_and_known_layers(tmp_path: Path):
    ws, state = stack(tmp_path)
    with pytest.raises(LookError) as caught:
        look(ws, state, LookRequest("atlas"))
    assert caught.value.code == "NO_POSITIONS"
    for layers in (("nissl",), ("fur",)):  # no Nissl for the synthetic atlas
        with pytest.raises(LookError) as caught:
            look(ws, state, LookRequest("atlas", positions_mm=(0.1,), atlas_layers=layers))
        assert caught.value.code == "UNKNOWN_LAYER"
    pictures = look(ws, state, LookRequest("atlas", positions_mm=(0.05, 0.1),
                                           atlas_layers=("template", "borders")))
    assert [p.extra["position_mm"] for p in pictures] == [0.05, 0.1]
    for picture in pictures:
        recipe = _recipe_ok(picture, "atlas")
        assert recipe["args"]["positions_mm"] == [picture.extra["position_mm"]]
        assert recipe["state"]["sections"] == {} and "view_angles" in recipe["state"]
        assert "layers template + borders" in picture.caption
    # The old name of the template is still read.
    assert look(ws, state, LookRequest("atlas", positions_mm=(0.1,), atlas_layers=("ara",)))


def test_overlay_keeps_the_placement_frame_for_the_layers(tmp_path: Path):
    ws, state = stack(tmp_path)
    with collecting() as notes:
        picture = look(ws, state, LookRequest("overlay", sections=("s2.png",)))[0]
    _recipe_ok(picture, "overlay")
    held = note_for(picture.image, notes)
    assert held is not None
    assert held.frame is not None and held.panel is not None  # layers still saved
    assert held.recipe == picture.recipe and held.caption == picture.caption
    assert "identity transform" in picture.caption and "atlas borders" in picture.caption
    record(state, "s2.png").position_mm = None
    with pytest.raises(LookError) as caught:
        look(ws, state, LookRequest("overlay", sections=("s2.png",)))
    assert caught.value.code == "NO_POSITION"
    with pytest.raises(LookError) as caught:
        look(ws, state, LookRequest("overlay", sections=("s1.png",), warp="bent"))
    assert caught.value.code == "BAD_WARP"


def test_overlay_atlas_images_get_an_opacity(tmp_path: Path):
    ws, state = stack(tmp_path)
    picture = look(ws, state, LookRequest("overlay", sections=("s0.png",),
                                          atlas_layers=("template", "borders")))[0]
    assert "images at 0.5" in picture.caption


def test_snapshot_changed_and_restored(tmp_path: Path):
    ws, state = stack(tmp_path)
    held = snapshot(state, ["s1.png"])
    assert not changed_since(state, held)
    moved = record(state, "s1.png")
    moved.position_mm = 0.2
    moved.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    assert changed_since(state, held)
    back = restored(state, held)
    assert record(back, "s1.png").position_mm == pytest.approx(0.15)
    assert record(back, "s1.png").transform is None
    assert moved.position_mm == 0.2  # the live state is untouched
    # A change to another section does not touch this picture.
    moved.position_mm, moved.transform = 0.15, None
    record(state, "s0.png").position_mm = 0.22
    assert not changed_since(state, held)
    # A section gone.
    state.slices = [s for s in state.slices if s.id != "s1.png"]
    assert changed_since(state, held)
    with pytest.raises(LookError) as caught:
        restored(state, held)
    assert caught.value.code == "UNKNOWN_SECTION"


def test_request_round_trips_through_recipe_args():
    request = LookRequest("overlay", sections=("a.tif",), positions_mm=(1.5,), channels=("DAPI",),
                          atlas_layers=("borders",), atlas_opacity=0.3, warp="none",
                          long_edge=640, zoom=(0.1, 0.2, 0.5, 0.6))
    assert LookRequest.from_args(json.loads(json.dumps(request.args()))) == request
    part = LookRequest("positioning", positions_mm=(2.0,), part=3)
    assert LookRequest.from_args(json.loads(json.dumps(part.args()))) == part
    assert LookRequest.from_args({"mode": "positioning"}).part is None  # an older recipe
