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
        id="s0.png", channel_recipe=None, damage_margin_mm=margin, atlas_image_note=note)


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
    ctx = SimpleNamespace(caption=lambda image, text: (captions.append(text), image)[1])
    frame = SimpleNamespace(channel=np.full((40, 60, 3), 200, np.uint8))
    assert _sorting_picture(ctx, frame, np.ones((40, 60), np.float32), "t") is None
    weights = np.ones((40, 60), np.float32)
    weights[5:15, 5:15] = 0.0
    picture = _sorting_picture(ctx, frame, weights, "t")
    assert isinstance(picture, Image.Image) and captions == ["t"]
