"""A job the ABBA connector prepared: its saved settings are the defaults of
every opening (``job.host_inputs.opening_spec``), on every door."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.doors.abba.contract import AbbaPrepareRequest
from langslice.doors.abba.prepare import prepare_job
from langslice.job.host_inputs import prepared_spec
from langslice.job.settings.spec import JobSpec
from tests.fakes import SlabAtlas

NAMES = ("s0.png", "s1.png", "s2.png")
POSITIONS = {"s0.png": 1.0, "s1.png": 2.0, "s2.png": 3.0}
_ATLAS = SlabAtlas()


def _prepared(root: Path) -> Path:
    folder = root / "snapshots"
    folder.mkdir(parents=True)
    for index, name in enumerate(NAMES):
        Image.fromarray(np.full((30, 40, 3), 60 + 30 * index, dtype=np.uint8)).save(
            folder / name)
    prepare_job(AbbaPrepareRequest.model_validate({
        "image_folder": str(folder), "pixel_size_um": 25, "positions_mm": POSITIONS,
        "angles_deg": {"pitch_deg": 1.0, "yaw_deg": 0.0},
        "registered": ["s2.png"], "existing_warp": [],
        "abba": {"project_key": "k", "atlas": "Allen Mouse V3p1", "sections": {}}}))
    return folder


def _init(folder: Path, *flags: str) -> dict[str, Any]:
    from langslice.doors.agent_cli.calls import init

    envelope = init(str(folder), list(flags), atlas_loader=lambda _n: _ATLAS)
    assert envelope.ok, envelope
    return envelope.result


def _saved_spec(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "langslice" / "job.json").read_text())["spec"]


def _state(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "langslice" / "state.json").read_text())


def test_init_continues_the_prepared_job_with_abbas_inputs(tmp_path: Path):
    folder = _prepared(tmp_path)
    result = _init(folder, "--tasks", "position,transform,nonlinear",
                   "--nonlinear-provider", "none", "--damaged", '{"s0.png": "torn"}')
    assert result["tasks"] == ["position", "transform", "nonlinear"]
    spec = _saved_spec(folder)
    # what the command gave wins; ABBA's inputs, freeze and angle estimate stay
    assert spec["tasks"] == ["position", "transform", "nonlinear"]
    assert spec["inputs"]["positions"] == POSITIONS
    assert spec["inputs"]["pixel_size_um"] == 25.0
    assert spec["inputs"]["angles"] == {"pitch": 1.0, "yaw": 0.0}
    assert spec["inputs"]["damaged"] == {"s0.png": "torn"}  # the command's own input
    assert spec["frozen"] == {"s2.png": ["orientation", "affine"]}
    assert spec["transform"]["angles"] is True
    state = _state(folder)
    placed = {row["id"]: row["position_mm"] for row in state["slices"]}
    assert placed == POSITIONS
    assert state["cutting_angles_deg"] == {"pitch": 1.0, "yaw": 0.0}
    held = {row["id"]: row.get("transform") for row in state["slices"]}
    assert held["s2.png"]["kind"] == "host"  # the user's ABBA registration


def test_an_explicit_empty_freeze_replaces_the_saved_one(tmp_path: Path):
    folder = _prepared(tmp_path)
    _init(folder, "--frozen", "{}")
    assert "frozen" not in _saved_spec(folder)
    assert _state(folder)["slices"][2].get("transform") is None


def test_cut_straight_freezes_the_angles_with_the_estimate_on(tmp_path: Path):
    """The generator's "cut straight" box beside ABBA's angle estimate."""
    spec = JobSpec(image_folder=str(tmp_path), frozen={"*": ["angles"]})
    spec = JobSpec.from_dict({**spec.to_dict(), "transform": {"angles": True}})
    assert spec.transform.angles and "angles" in spec.frozen_for("any.png")
    folder = _prepared(tmp_path)
    _init(folder, "--frozen", '{"*": ["angles"]}')
    assert _saved_spec(folder)["frozen"] == {"*": ["angles"]}
    assert _saved_spec(folder)["transform"]["angles"] is True


def test_a_second_init_resumes_the_abba_job(tmp_path: Path):
    folder = _prepared(tmp_path)
    _init(folder)
    first = _state(folder)
    result = _init(folder)
    assert _saved_spec(folder)["frozen"] == {"s2.png": ["orientation", "affine"]}
    assert _state(folder)["slices"] == first["slices"]
    assert result["sections"] == list(NAMES)


def test_the_library_and_mcp_open_a_prepared_job_with_its_inputs(tmp_path: Path):
    import langslice
    from langslice.doors.mcp.server import open_folder

    folder = _prepared(tmp_path / "library")
    job = langslice.create_job(str(folder), atlas_loader=lambda _n: _ATLAS)
    try:
        assert job.job.spec.inputs["positions"] == POSITIONS
        assert job.job.spec.frozen == {"s2.png": ["orientation", "affine"]}
    finally:
        job.close()

    folder = _prepared(tmp_path / "mcp")
    session = open_folder(str(folder), lambda path: JobSpec(image_folder=path),
                          lambda _n: _ATLAS)
    try:
        assert session.spec.inputs["positions"] == POSITIONS
        assert session.spec.transform.angles
        assert "s2.png" in session.spec.frozen
        positions = {s.id: s.position_mm for s in session.state.in_order()}
        assert positions == POSITIONS
    finally:
        session.close()


def test_prepared_spec_takes_the_saved_values_where_the_opening_gives_defaults():
    saved = JobSpec.from_dict({
        "image_folder": "/snapshots", "inputs": {"pixel_size_um": 25.0},
        "host_preprocessing": {"mode": "auto"}, "frozen": {"a.png": ["affine"]},
        "transform": {"angles": True}, "position": {"thickness_um": 40},
        "resume": False}).to_dict()
    asked = JobSpec(image_folder="/here", tasks=["position"], inputs={"pixel_size_um": 10.0})
    merged = prepared_spec(saved, asked, frozen_given=False)
    assert merged.tasks == ["position"]  # given
    assert merged.transform.angles and merged.position.thickness_um == 40  # saved
    assert merged.inputs == {"pixel_size_um": 25.0}  # the host's, always
    assert merged.host_preprocessing == {"mode": "auto"}
    assert merged.frozen == {"a.png": ["affine"]}
    assert merged.image_folder == "/here" and merged.resume
    assert prepared_spec(saved, asked, frozen_given=True).frozen == {}
    # opening twice is opening once
    again = prepared_spec(saved, merged, frozen_given=False)
    assert again.to_dict() == merged.to_dict()


def test_a_job_that_is_not_abbas_is_opened_as_given(tmp_path: Path):
    from langslice.job.host_inputs import opening_spec
    from langslice.job.layout import JobLayout, write_job_file

    layout = JobLayout(tmp_path / "langslice", tmp_path)
    layout.ensure()
    spec = JobSpec(image_folder=str(tmp_path))
    assert opening_spec(layout.folder, spec, frozen_given=False) is spec
    write_job_file(layout, spec={**spec.to_dict(), "tasks": ["nonlinear"]},
                   host={"format": 2, "kind": "host"})
    assert opening_spec(layout.folder, spec, frozen_given=False) is spec


@pytest.mark.parametrize("flags", [(), ("--no-resume",)])
def test_starting_an_abba_job_over_keeps_abbas_inputs(tmp_path: Path, flags):
    folder = _prepared(tmp_path)
    _init(folder)
    _init(folder, *flags)
    placed = {row["id"]: row["position_mm"] for row in _state(folder)["slices"]}
    assert placed == POSITIONS
