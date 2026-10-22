"""What every fit reply says the same way: the mask shape, the damage margin, the pictures'
captions, the default Nissl request, and a fit that fitted nothing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from langslice.core.atlas.images import AUGMENTED_ATLAS, nissl_note
from langslice.doors.tools.fitting import nothing_fitted
from langslice.engines import _blocks


def test_a_fit_that_fitted_nothing_leads_with_the_reason_every_section_shares():
    row = {"status": "error", "error": "HAS_DEFORMATION", "message": "holds a deformation."}
    one = nothing_fitted([{"id": "a", **row}])
    assert one == {"error": "HAS_DEFORMATION", "message": "holds a deformation."}
    two = nothing_fitted([{"id": "a", **row}, {"id": "b", **row}])
    assert two["error"] == "HAS_DEFORMATION" and "holds a deformation." in two["message"]
    mixed = nothing_fitted([{"id": "a", **row}, {"id": "b", "error": "FROZEN"}])
    assert mixed == {"error": "NOTHING_FITTED"}
    assert nothing_fitted([]) == {"error": "NOTHING_FITTED"}


def test_the_default_nissl_request_is_said_honoured_where_the_template_is_a_nissl():
    augmented = SimpleNamespace(atlas_name=AUGMENTED_ATLAS, additional_references={})
    assert "honoured" in (nissl_note(augmented, "nissl") or "")
    assert nissl_note(augmented, "template") is None
    other = SimpleNamespace(atlas_name="other", additional_references={})
    assert nissl_note(other, "nissl") is None  # an atlas without one is not claimed to have one


def _frame(margin: float | None, note: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="s0.png", section_image_ref=None, damage_margin_mm=margin, atlas_image_note=note)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(section=lambda _sid: SimpleNamespace(damaged_regions=["TH"]))


def test_an_engine_inputs_block_has_the_native_shapes():
    block = _blocks.inputs_block(_ctx(), _frame(0.1), section_image="preprocessed",
                                 atlas_image="nissl", resolved="template", mask=None,
                                 brightness="x")
    assert block["mask"] == {"name": "none", "kind": None, "stale": False,
                             "earlier_registration": False}
    assert block["damage"] == {"regions": ["TH"], "margin_mm": 0.1, "side": "atlas"}
    assert "restrict_to" not in block and "atlas_image_note" not in block
    mask = SimpleNamespace(name="hole", kind="em", stale=False, earlier_registration=True)
    named = _blocks.inputs_block(_ctx(), _frame(0.1, "honoured"), section_image="dapi",
                                 atlas_image="nissl", resolved="template", mask=mask,
                                 brightness="x")
    assert named["mask"]["name"] == "hole" and named["mask"]["earlier_registration"] is True
    assert named["atlas_image_note"] == "honoured"


def test_a_sorting_that_set_nothing_aside_draws_no_picture():
    from langslice.engines.stalign.plugin import _sorting_picture

    captions: list[str] = []
    shown: list[object] = []
    ctx = SimpleNamespace(caption=lambda image, text, sections=(): (
        captions.append(text), shown.append(list(sections)), image)[2])
    frame = SimpleNamespace(id="s0.png", section_image=np.full((40, 60, 3), 200, np.uint8))
    assert _sorting_picture(ctx, frame, np.ones((40, 60), np.float32), "t") is None
    weights = np.ones((40, 60), np.float32)
    weights[5:15, 5:15] = 0.0
    picture = _sorting_picture(ctx, frame, weights, "t")
    assert isinstance(picture, Image.Image) and captions == ["t"]
    assert shown == [["s0.png"]]  # the picture names its section


def test_a_notice_names_each_flag_once_with_its_regions():
    from langslice.plugin_api import flag_words

    summary = {"flags": [
        {"code": "FOLDS", "fold_fraction": 0.3},
        {"code": "DISPLACEMENT_OUTSIZED", "max_mm": 3.0},
        {"code": "REGION_COMPRESSED", "region": "CA1", "area_ratio": 0.4},
        {"code": "REGION_EXPANDED", "region": "VL", "area_ratio": 2.5},
        {"code": "REGION_COMPRESSED", "region": "DG", "area_ratio": 0.3},
        {"code": "REGION_COMPRESSED", "region": "CA1", "area_ratio": 0.4},
    ], "more_flags": 2}
    assert flag_words(summary) == ("FOLDS, DISPLACEMENT_OUTSIZED, REGION_COMPRESSED (CA1, DG), "
                                   "REGION_EXPANDED (VL), 2 more")
    assert flag_words({"flags": []}) == "none"


def test_two_calls_at_once_get_folders_of_their_own(tmp_path):
    ctx = SimpleNamespace(work_dir=lambda: tmp_path)
    first, second = _blocks.call_folder(ctx, "warp"), _blocks.call_folder(ctx, "warp")  # pyright: ignore[reportArgumentType]
    assert first != second and first.is_dir() and second.is_dir()
