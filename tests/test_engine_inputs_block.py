"""Every native fit tool's reply carries ``inputs`` and ``estimated`` per section
and the call's ``settings``; the record's provenance holds the same blocks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.linear_tool_helpers import tool_named as _tool
from tests.test_set_mask_tool import ID, SyntheticAtlas, _setup

pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")

INPUT_KEYS = {"section_image", "section_recipe", "atlas_image", "atlas_image_resolved",
              "mask", "damage", "restrict_to", "brightness"}


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


@pytest.mark.parametrize("name", ["elastix_affine", "fireants_syn"])
def test_every_fit_reply_names_what_it_used(tmp_path: Path, atlas, name: str):
    state, _, box = _setup(tmp_path, atlas)
    kwargs = {"detail": "coarse"} if name == "fireants_syn" else {}
    done = _tool(box, name)([ID], view=False, **kwargs)
    assert done["status"] == "ok", done
    assert "settings" in done
    (row,) = done["results"]
    assert set(row["inputs"]) == INPUT_KEYS
    assert row["inputs"]["damage"]["side"] == "atlas"
    assert "estimated" in row and row["estimated"] is None
    assert "quality" not in str(row).lower()
