"""``abba.prepare``: an ABBA session's snapshots become a prepared job.

The job folder beside the snapshots gets ``job.json`` (ABBA's inputs, the
registrations made in ABBA frozen, the angle estimate on, the session's
facts under ``host.abba``) and the reference card; no checkpoint, no index
entry, no prompt."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from langslice.doors.abba.contract import AbbaPrepareRequest
from langslice.doors.abba.prepare import prepare_job
from langslice.hosts.service import run_stdio
from langslice.job.layout import JobLayout, read_job_file

NAMES = ("s0.png", "s1.png", "s2.png")


def _snapshots(root: Path) -> Path:
    folder = root / "abba" / "0123456789ab"
    folder.mkdir(parents=True)
    for index, name in enumerate(NAMES):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            folder / name)
    return folder


def _params(folder: Path, **change: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "image_folder": str(folder), "pixel_size_um": 25,
        "positions_mm": {"s0.png": 3.0, "s1.png": 1.0, "s2.png": 2.0},
        "z_offset_mm": 5.7, "angles_deg": {"pitch_deg": 1.5, "yaw_deg": -0.5},
        "existing_warp": ["s2.png"], "registered": ["s1.png", "s2.png"],
        "abba": {"project_key": "0123456789ab", "atlas": "Adult Mouse Brain - Allen V3p1",
                 "sections": {"s0.png": "slice A", "s1.png": "slice B", "s2.png": "slice C"},
                 "thickness_um": 40, "interval_um": 120},
    }
    params.update(change)
    return params


def test_the_prepared_job_holds_abbas_inputs_and_facts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = _snapshots(tmp_path)
    job_folder = prepare_job(AbbaPrepareRequest.model_validate(_params(folder)))
    assert job_folder == folder / "langslice"
    record = read_job_file(JobLayout(job_folder))
    assert record is not None
    spec = record["spec"]
    assert spec["inputs"] == {
        "positions": {"s0.png": 3.0, "s1.png": 1.0, "s2.png": 2.0},
        "order": ["s1.png", "s2.png", "s0.png"], "pixel_size_um": 25.0,
        "angles": {"pitch": 1.5, "yaw": -0.5}}
    # what the user registered in ABBA is theirs; a section's own warp too
    assert spec["frozen"] == {"s1.png": ["orientation", "affine"],
                              "s2.png": ["orientation", "affine", "deformation"]}
    assert spec["transform"]["angles"] is True
    assert spec["resume"] is False and spec["host_preprocessing"] is None
    assert spec["atlas"] == "allen_mouse_25um" and spec["plane"] == "coronal"
    host = record["host"]
    assert host["format"] == 3 and host["kind"] == "abba"
    abba = host["abba"]
    assert abba["project_key"] == "0123456789ab"
    assert abba["sections"]["s1.png"] == "slice B"
    assert abba["registered"] == ["s1.png", "s2.png"] and abba["existing_warp"] == ["s2.png"]
    assert abba["angles_deg"] == {"pitch_deg": 1.5, "yaw_deg": -0.5}
    assert abba["z_offset_mm"] == 5.7 and abba["pixel_size_um"] == 25.0
    assert abba["thickness_um"] == 40 and abba["interval_um"] == 120
    assert abba["prepared_at"]
    # a prepared job: no checkpoint, no index entry, no prompt; the card is there
    assert not (job_folder / "state.json").exists()
    assert not (tmp_path / "home" / ".langslice" / "jobs").exists()
    assert not (job_folder / "prompt.txt").exists()
    assert (job_folder / "AGENTS.md").is_file() and (job_folder / "CLAUDE.md").is_file()


def test_preparing_again_before_the_job_opened_replaces_the_settings(tmp_path: Path):
    folder = _snapshots(tmp_path)
    prepare_job(AbbaPrepareRequest.model_validate(_params(folder)))
    job_folder = prepare_job(AbbaPrepareRequest.model_validate(
        _params(folder, registered=[], existing_warp=[])))
    record = read_job_file(JobLayout(job_folder))
    assert record is not None and "frozen" not in record["spec"]


@pytest.mark.parametrize("change, message", [
    ({"positions_mm": {"s0.png": 1.0}}, "every exported snapshot"),
    ({"existing_warp": ["missing.png"]}, "existing_warp"),
    ({"registered": ["missing.png"]}, "registered"),
    ({"abba": {"project_key": "k", "atlas": "a", "sections": {"missing.png": "x"}}},
     "abba.sections"),
    ({"channel_names": ["DAPI", "DAPI"]}, "distinct"),
])
def test_names_that_are_not_the_snapshots_are_refused(tmp_path: Path, change, message):
    folder = _snapshots(tmp_path)
    with pytest.raises(ValueError, match=message):
        prepare_job(AbbaPrepareRequest.model_validate(_params(folder, **change)))
    assert not (folder / "langslice").exists()


@pytest.mark.parametrize("change", [
    {"pixel_size_um": 0},
    {"positions_mm": {"s0.png": float("nan"), "s1.png": 1.0, "s2.png": 2.0}},
    {"angles_deg": {"pitch_deg": float("inf"), "yaw_deg": 0.0}},
    {"angles_deg": {"pitch": 1.0, "yaw": 0.0}},
    {"z_offset_mm": float("inf")},
    {"spec": {"tasks": ["position"]}},
    {"preprocessing": {"mode": "auto"}},
    {"abba": {"project_key": "", "atlas": "a", "sections": {}}},
    {"abba": {"project_key": "k", "atlas": "a", "sections": {}, "thickness_um": -1}},
])
def test_a_malformed_request_is_refused_before_anything_is_written(tmp_path: Path, change):
    folder = _snapshots(tmp_path)
    with pytest.raises(ValidationError):
        AbbaPrepareRequest.model_validate(_params(folder, **change))


def test_a_job_that_already_ran_is_not_prepared_again(tmp_path: Path):
    folder = _snapshots(tmp_path)
    job_folder = prepare_job(AbbaPrepareRequest.model_validate(_params(folder)))
    (job_folder / "state.json").write_text("{}")
    before = (job_folder / "job.json").read_text()
    with pytest.raises(ValueError, match="already ran"):
        prepare_job(AbbaPrepareRequest.model_validate(_params(folder)))
    assert (job_folder / "job.json").read_text() == before


def test_the_service_answers_prepare_with_the_job_folder(tmp_path: Path):
    folder = _snapshots(tmp_path)
    line = json.dumps({"id": "p", "method": "abba.prepare", "params": _params(folder)})
    output = io.StringIO()
    run_stdio(input_stream=io.StringIO(line + "\n"), output_stream=output)
    [message] = [json.loads(text) for text in output.getvalue().splitlines()]
    assert message == {"id": "p", "type": "result",
                       "result": {"job_folder": str(folder / "langslice")}}


def test_a_section_keeping_the_users_warp_is_refused_by_fireants_syn(tmp_path, monkeypatch):
    """A slice with its own warp in ABBA reaches the job as a frozen
    deformation; the deformable fit refuses it."""
    import dataclasses

    import langslice
    from langslice.doors.jobs import create
    from langslice.job.settings.spec import NonlinearSpec
    from tests.synthetic_stack import (
        ID0,
        PIXEL_SIZE_UM,
        atlas_loader,
        full_spec,
        write_sections,
    )

    pytest.importorskip("fireants", reason="fireants_syn needs FireANTs")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = tmp_path / "stack"
    write_sections(folder)
    spec = dataclasses.replace(
        full_spec(folder), nonlinear=NonlinearSpec(provider="none"),
        inputs={"pixel_size_um": PIXEL_SIZE_UM}, frozen={ID0: ["deformation"]})
    create(spec, atlas_loader=atlas_loader()).close()
    job = langslice.open_job(str(folder), atlas_loader=atlas_loader())
    try:
        job.position_sections(sections=[{"id": ID0, "position_mm": 0.1}], view=False)
        job.interactive_transform(sections=[{"id": ID0, "rotation_deg": 0.0}], view=False)
        result = job.fireants_syn(sections=[ID0], view=False)
        assert "FROZEN" in json.dumps(result)
        assert job.state.by_id(ID0).deformation is None
        refusal = job.job.frozen_refusal([ID0], "deformation")
        assert refusal is not None and refusal.code == "FROZEN"
    finally:
        job.close()
