"""The ``inputs`` / ``estimated`` blocks and the resolved choice a fit echoes."""

from __future__ import annotations

from langslice.core import deformation
from langslice.core.engine_inputs import (
    NATIVE_BRIGHTNESS,
    EstimatedBlock,
    InputsBlock,
    damage_block,
    mask_block,
)


def test_the_inputs_block_names_what_was_used():
    image = {"name": "preprocessed", "kind": "made",
             "file": "sections/s0/images/preprocessed.png", "digest": "ab12cd34"}
    block = InputsBlock(
        section_image=image, atlas_image="nissl",
        atlas_image_resolved="ccfv3augmented_nissl", mask=mask_block(),
        damage=damage_block(["TH"], 0.1), restrict_to=["STR"])
    assert block.payload() == {
        "section_image": image, "atlas_image": "nissl",
        "atlas_image_resolved": "ccfv3augmented_nissl",
        "mask": {"name": "none", "kind": None, "stale": False, "earlier_registration": False},
        "damage": {"regions": ["TH"], "margin_mm": 0.1, "side": "atlas"},
        "restrict_to": ["STR"], "brightness": NATIVE_BRIGHTNESS}
    raw = {"name": "dapi", "kind": "raw", "file": "s0.tif", "digest": None}
    named = InputsBlock(raw, "template", "template",
                        mask_block("hole", "em"), damage_block([], 0.1))
    assert named.payload()["mask"] == {"name": "hole", "kind": "em", "stale": False,
                                       "earlier_registration": False}
    assert named.payload()["restrict_to"] == []


def test_the_estimated_block():
    assert EstimatedBlock().payload() == {"sorting": None, "left_out_fraction": None}
    saved = EstimatedBlock("saved with the deformation; make_mask source fit", 0.07)
    assert saved.payload()["left_out_fraction"] == 0.07


def test_a_choice_carries_the_smoothness_and_the_mask():
    choice = deformation.Choice(fit_section="fit", fit_atlas="nissl", smoothness_um=120.0,
                                mask="hole")
    assert choice.echo() == {"smoothness_um": 120.0, "mask": "hole", "fit_section": "fit",
                             "fit_atlas": "nissl"}
    settings = choice.settings((), ())
    assert settings.smoothness_um == 120.0 and settings.atlas_image == "nissl"
    assert deformation.Choice("fit", "template").smoothness_um == 80.0
    used = deformation.engine_settings(settings)
    assert used["update_sigma_mm"] == 0.12 and used["total_sigma_mm"] == 0.03
