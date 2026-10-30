"""The fit tools' per-section rows as replies give them: the keys and their
order each kind of row had when the rows were dicts (``tools.deformable_rows``,
``tools.affine_fit.affine_row_to_json``, ``tools.refusal.refused_row``)."""

from __future__ import annotations

from langslice.core.refusal import RefusalCode, Refused
from langslice.tools.affine_fit import AffineRow, affine_row_to_json
from langslice.tools.deformable_rows import (
    AppliedFit,
    DeformableRow,
    deformable_row_to_json,
)

SETTINGS = {"section_image": "preprocessed", "atlas_image": "nissl",
            "smoothness_um": 80.0, "mask": "none"}
APPLIED = AppliedFit(engine_settings={"metric": "mi"},
                     numbers={"displacement_mm": {"max": 0.1, "median": 0.05},
                              "fold_fraction": 0.0, "flags": []},
                     runtime_s=1.5, cached=False, gpu={})


def test_a_section_refused_before_its_fit_names_itself_first():
    frozen = Refused(RefusalCode.FROZEN, properties=["deformation"], message="m")
    row = deformable_row_to_json(DeformableRow("a.png", refusal=frozen))
    assert list(row) == ["id", "status", "error", "properties", "message"]


def test_a_refusal_while_the_fit_is_set_up_stands_after_its_status():
    refused = Refused(RefusalCode.STALE_MASK, message="m")
    row = deformable_row_to_json(DeformableRow("a.png", SETTINGS, refusal=refused))
    assert list(row) == ["id", "status", "settings", "error", "message"]
    assert row["status"] == "error"


def test_a_refusal_after_the_fit_ran_stands_after_its_settings():
    stale = Refused(RefusalCode.STALE_INPUT, message="m")
    row = deformable_row_to_json(DeformableRow(
        "a.png", SETTINGS, trace_regions={"include": [], "exclude": ["HY"]},
        refusal=stale, queued=True))
    assert list(row) == ["id", "trace_regions", "settings", "status", "error", "message"]


def test_a_fit_whose_record_was_not_saved_keeps_its_numbers():
    failed = Refused(RefusalCode.RECORD_WRITE_FAILED, message="disk full")
    row = deformable_row_to_json(DeformableRow(
        "a.png", SETTINGS, refusal=failed, queued=True, applied=APPLIED))
    assert list(row) == ["id", "settings", "status", "engine_settings", "displacement_mm",
                         "fold_fraction", "flags", "runtime_s", "cached", "error", "message"]
    assert row["status"] == "error"


def test_an_applied_fit_ends_with_what_was_written():
    row = deformable_row_to_json(DeformableRow(
        "a.png", SETTINGS, queued=True, applied=APPLIED, steps=2, image_indexes=(0,)))
    assert list(row)[-4:] == ["cached", "written", "steps", "image_indexes"]
    assert row["status"] == "ok" and row["written"] is True


def test_the_core_fits_refusal_keeps_the_section_where_it_named_it():
    refused = Refused(RefusalCode.NO_POSITION, id="a.png", message="m")
    assert list(affine_row_to_json(AffineRow("a.png", refusal=refused))) == [
        "status", "error", "id", "message"]
    stale = Refused(RefusalCode.STALE_INPUT, message="m")
    assert list(affine_row_to_json(AffineRow("a.png", refusal=stale))) == [
        "id", "status", "error", "message"]
