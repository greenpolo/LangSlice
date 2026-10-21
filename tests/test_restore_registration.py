"""Every registration a section holds is numbered, and one section goes back to one.

A section's registration is its orientation, transform and deformation
(``langslice.job.registrations``). Each write that leaves a new one lists it
under the next number (replies and status rows name it); ``restore_registration``
puts an earlier one back on that section alone, as one undo step, refusing a
frozen property, an unknown number and a deformation fitted at another
position.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import langslice
from langslice.doors.cli.jobcli import main
from langslice.job import registrations as regs
from langslice.job.background import DONE, Work, notice_text
from tests.cli_child import install
from tests.synthetic_stack import ID0, ID1, PIXEL_SIZE_UM, atlas_loader, write_sections

IDENTITY = {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0, "translate_x_mm": 0.0,
            "translate_y_mm": 0.0}


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("registrations") / "stack"
    write_sections(images)
    return images


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(list(argv))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1, lines
    return int(code or 0), json.loads(lines[0])


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           capsys: pytest.CaptureFixture[str]) -> Path:
    """Two placed sections with identity transforms, and their job."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    install(atlas_loader(), monkeypatch.setattr)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    code, envelope = cli(capsys, str(folder), "init", "--tasks",
                         "position,transform,nonlinear", "--image-provider", "none",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM))
    assert code == 0, envelope
    code, envelope = cli(capsys, str(folder), "position_sections", "--sections", json.dumps(
        [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15}]),
        "--view", "false")
    assert code == 0, envelope
    code, envelope = cli(capsys, str(folder), "interactive_transform", "--sections",
                         json.dumps([{"id": ID0, **IDENTITY}, {"id": ID1, **IDENTITY}]),
                         "--view", "false")
    assert code == 0, envelope
    return folder


def _needs_engine() -> None:
    from langslice.ops.deformable import engine_ready

    if not engine_ready():
        pytest.skip("fireants_syn needs FireANTs")


def rows(job: Any) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in job.status()["rows"]}


def test_writes_number_registrations_and_one_section_goes_back(capsys, images):
    _needs_engine()
    job = langslice.open_job(images)
    # The hand transforms of the fixture are each section's registration 1.
    assert rows(job)[ID0]["registration"] == 1 and rows(job)[ID1]["registration"] == 1
    # The reply's row names the registration the write left.
    first = job.fireants_syn(sections=[ID0], view=False)
    assert first["changed"][0]["registration"] == 2 and "registration" not in first
    fitted = job.state.resolve(ID0).deformation["key"]
    # Another section's write comes between, then another fit replaces the first.
    turned = job.interactive_transform(sections=[{"id": ID1, **IDENTITY, "rotation_deg": 3.0}],
                                       view=False)
    assert turned["changed"][0]["registration"] == 2
    again = job.fireants_syn(sections=[ID0], smoothness_um=40.0, view=False)
    assert again["changed"][0]["registration"] == 3
    assert job.state.resolve(ID0).deformation["key"] != fitted

    listed = job.restore_registration(section=ID0)
    assert listed["held"] == 3
    assert [(entry["registration"], entry["by"]) for entry in listed["registrations"]] == [
        (1, "interactive_transform"), (2, "fireants_syn"), (3, "fireants_syn")]
    assert listed["registrations"][1]["deformation_steps"] == 1
    assert listed["registrations"][2]["held"] is True

    depth = len(job.job.undo_stack)
    restored = job.restore_registration(section=ID0, registration=2)
    assert restored["status"] == "ok" and restored["written"] is True
    assert restored["registration"] == 2 and restored["replaced"] == 3
    assert restored["changed"][0]["registration"] == 2
    assert job.state.resolve(ID0).deformation["key"] == fitted
    assert job.state.resolve(ID1).transform["physical"]["rotation_deg"] == pytest.approx(3.0)
    assert len(job.job.undo_stack) == depth + 1
    assert rows(job)[ID0]["registration"] == 2
    # The same one again writes nothing; undo takes the restore back.
    assert job.restore_registration(section=ID0, registration=2)["written"] is False
    job.undo()
    assert rows(job)[ID0]["registration"] == 3
    # The list on disk is the job's, one line per registration.
    lines = (images / "langslice" / "sections" / Path(ID0).stem / regs.FILE).read_text()
    assert len(lines.splitlines()) == 3
    job.close()


def test_a_transform_alone_goes_back_and_the_numbers_are_refused_with_a_fix(capsys, images):
    job = langslice.open_job(images)
    job.interactive_transform(sections=[{"id": ID0, **IDENTITY, "scale_x": 1.1}], view=False)
    unknown = job.restore_registration(section=ID0, registration=9)
    assert unknown["error"] == "UNKNOWN_REGISTRATION" and unknown["registrations"] == [1, 2]
    assert "list" in unknown["message"]
    back = job.restore_registration(section=ID0, registration=1)
    assert back["written"] is True
    assert job.state.resolve(ID0).transform["physical"]["scale_x"] == pytest.approx(1.0)
    code, envelope = cli(capsys, str(images), "restore_registration", "--section", ID0,
                         "--registration", "9")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_REGISTRATION"
    assert "restore_registration" in envelope["error"]["fix"]
    job.close()


def test_a_frozen_property_and_a_deformation_fitted_elsewhere_are_refused(capsys, images):
    _needs_engine()
    job = langslice.open_job(images)
    job.fireants_syn(sections=[ID0], view=False)                      # registration 2
    job.position_sections(sections=[{"id": ID0, "position_mm": 0.12}], view=False)
    assert job.state.resolve(ID0).deformation is None                # made stale
    elsewhere = job.restore_registration(section=ID0, registration=2)
    assert elsewhere["error"] == "REGISTRATION_ELSEWHERE"
    assert "0.1 mm" in elsewhere["message"] and "0.12 mm" in elsewhere["message"]
    listed = job.restore_registration(section=ID0)["registrations"]
    assert "not_here" in listed[1] and "not_here" not in listed[0]
    job.position_sections(sections=[{"id": ID0, "position_mm": 0.1}], view=False)
    job.job.spec = dataclasses.replace(job.job.spec, frozen={ID0: ["deformation"]})
    frozen = job.restore_registration(section=ID0, registration=2)
    assert frozen["error"] == "FROZEN" and frozen["properties"] == ["deformation"]
    assert job.state.resolve(ID0).deformation is None
    job.close()


def test_a_registration_held_before_the_list_is_listed_when_it_is_replaced(capsys, images):
    """A job folder older than the list: the registration a section holds is
    listed, by what its content says made it, when a write replaces it."""
    _needs_engine()
    job = langslice.open_job(images)
    job.fireants_syn(sections=[ID0], view=False)
    store = images / "langslice" / "sections" / Path(ID0).stem / regs.FILE
    store.unlink()
    job.close()
    job = langslice.open_job(images)
    assert rows(job)[ID0]["registration"] is None
    job.fireants_syn(sections=[ID0], smoothness_um=40.0, view=False)
    entries = [json.loads(line) for line in store.read_text().splitlines()]
    assert [(entry["id"], entry["by"], entry["at"] is None) for entry in entries] == [
        (1, "fireants_syn", True), (2, "fireants_syn", False)]
    assert job.restore_registration(section=ID0, registration=1)["written"] is True
    job.close()


def test_a_reply_without_the_section_s_row_names_its_registration():
    """A plugin tool's reply has its own rows: the registration its write left
    is named beside them; a section whose status row the reply shows is not."""
    from types import SimpleNamespace

    from langslice.doors.tools.toolbox import registrations_named

    job = SimpleNamespace(registered_since=lambda mark: {ID0: 3, ID1: 5})
    reply = registrations_named(job, {"rows": [{"id": ID0, "fitted": "x"}],  # pyright: ignore[reportArgumentType]
                                      "changed": [{"id": ID1, "registration": 5}]}, 0)
    assert reply["registration"] == {ID0: 3}
    assert registrations_named(job, {"registration": 2}, 0) == {"registration": 2}  # pyright: ignore[reportArgumentType]


def test_a_background_notice_names_the_registration_it_left():
    work = Work(id="w1", kind="stalign_lddmm", sections=("s0.png",), status=DONE,
                registrations={"s0.png": 4}, pictures=[7])
    assert notice_text(work, "STalign bend.") == (
        "w1 stalign_lddmm of s0.png finished: STalign bend. Now registration 4 of s0.png. "
        "Pictures #7.")
