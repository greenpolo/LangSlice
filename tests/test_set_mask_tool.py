"""``set_mask``: em and fit sources, delete, stale after a write, the masks a fit
takes by name, and ``HAS_DEFORMATION`` from ``elastix_affine``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from langslice.agent.engine import build_context
from langslice.core.section_masks import registration_digest
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from langslice.ops import masks as ops_masks
from langslice.ops.inputs import section_inputs
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section
from tests.linear_tool_helpers import tool_named as _tool

pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")

ID = "s0.png"


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _setup(folder: Path, atlas: SyntheticAtlas) -> tuple[Any, Any, Any]:
    image, _ = render_section(atlas, SMOOTH_FIELD())
    image.save(folder / ID)
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   tasks=["transform", "nonlinear"], inputs={"pixel_size_um": 25.0},
                   transform=TransformSpec(angles=True),
                   nonlinear=NonlinearSpec(provider="none"))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    record = state.slices[0]
    record.position_mm = 0.1
    record.transform = {"kind": "interactive", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "calibration": {"section_um_per_px": 25.0, "source": "host"}}
    return state, ctx, build_tools(state, ctx, spec)


def test_the_digest_a_mask_stores_is_the_fit_inputs_digest(tmp_path: Path, atlas):
    state, _, _ = _setup(tmp_path, atlas)
    record = state.slices[0]
    assert registration_digest(state, record) == section_inputs(state, record, deformation=True)


def test_an_em_mask_is_made_listed_stale_after_a_write_and_deleted(tmp_path: Path, atlas):
    state, _, box = _setup(tmp_path, atlas)
    tool = _tool(box, "set_mask")
    made = tool(ID, "tissue", view=False)
    assert made["status"] == "ok", made
    assert made["source"] == "em" and made["atlas_image"] == "template"
    assert 0.0 <= made["left_out"] <= 1.0
    assert len(box.job.undo_stack) == 1
    row = _tool(box, "status")()["rows"][0]
    assert row["masks"] == [{"name": "tissue", "kind": "em", "stale": False}]
    assert (tmp_path / "langslice" / "sections" / "s0" / "masks" / "tissue.npz").exists()
    # A write to the section's placement makes it stale; a fit refuses it.
    _tool(box, "interactive_transform")([{"id": ID, "rotation_deg": 3.0}], view=False)
    assert _tool(box, "status")()["rows"][0]["masks"][0]["stale"] is True
    fit = _tool(box, "elastix_affine")([ID], mask="tissue", view=False)
    assert fit["results"][0]["error"] == "STALE_MASK"
    # Made again it is fresh, and the reply of a fit names it.
    tool(ID, "tissue", view=False)
    fit = _tool(box, "elastix_affine")([ID], mask="tissue", view=False)
    assert fit["status"] == "ok", fit
    assert fit["results"][0]["inputs"]["mask"] == {"name": "tissue", "kind": "em",
                                                    "stale": False}
    assert tool(ID, "tissue", delete=True)["deleted"] is True
    assert tool(ID, "tissue", delete=True)["error"] == "UNKNOWN_MASK"
    assert tool(ID, "bad name!", view=False)["error"] == "BAD_ARGS"
    assert tool(ID, "none", view=False)["error"] == "BAD_ARGS"
    assert tool(ID, "m", channel="nope", view=False)["error"] == "UNKNOWN_CHANNEL"
    assert tool(ID, "m", sorting={"bogus": 1}, view=False)["error"] == "BAD_ARGS"


def test_the_mask_is_drawn_on_the_section_and_look_outlines_it(tmp_path: Path, atlas):
    _, _, box = _setup(tmp_path, atlas)
    made = _tool(box, "set_mask")(ID, "tissue")
    assert made["status"] == "ok", made
    assert len(made["pictures"]) == 1
    looked = _tool(box, "look")("section", sections=[ID], masks=["tissue"])
    assert looked["status"] == "ok" and looked["masks"][0]["name"] == "tissue"
    assert _tool(box, "look")("section", sections=[ID], masks=["nope"])["error"] == "UNKNOWN_MASK"


def test_the_fit_source_needs_a_deformation_that_saved_a_sorting(tmp_path: Path, atlas):
    state, _, box = _setup(tmp_path, atlas)
    tool = _tool(box, "set_mask")
    assert tool(ID, "kept", source="fit", view=False)["error"] == "NO_DEFORMATION"
    done = _tool(box, "fireants_syn")([ID], view=False, detail="coarse")
    assert done["status"] == "ok", done
    # FireANTs sorts nothing by itself now.
    assert tool(ID, "kept", source="fit", view=False)["error"] == "NO_SORTING"


def test_fireants_takes_an_em_mask_by_name_and_names_it(tmp_path: Path, atlas):
    state, _, box = _setup(tmp_path, atlas)
    _tool(box, "set_mask")(ID, "tissue", view=False)
    done = _tool(box, "fireants_syn")([ID], mask="tissue", view=False, detail="coarse")
    assert done["status"] == "ok", done
    row = done["results"][0]
    assert row["inputs"]["mask"]["name"] == "tissue" and done["settings"]["mask"] == "tissue"
    assert _tool(box, "fireants_syn")([ID], mask="nope")["error"] == "UNKNOWN_MASK"


def test_a_linear_refit_on_a_warped_section_is_refused(tmp_path: Path, atlas):
    state, _, box = _setup(tmp_path, atlas)
    _tool(box, "fireants_syn")([ID], view=False, detail="coarse")
    done = _tool(box, "elastix_affine")([ID], view=False)
    assert done["status"] == "error"
    assert done["results"][0]["error"] == "HAS_DEFORMATION"
    assert state.slices[0].deformation is not None


def test_set_mask_is_off_where_the_job_has_no_fit(tmp_path: Path, atlas):
    assert ops_masks.NO_MASK == "none"
