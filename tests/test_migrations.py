"""The table of migrations that reads old job folders as the current format."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import langslice
from langslice.doors.agent.engine import build_context
from langslice.job.job import Job
from langslice.job.migrations import STATE_MIGRATIONS, FolderMigration
from langslice.job.settings.spec import JobSpec
from langslice.job.state_file import STATE_FORMAT_VERSION, current_state
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def test_the_table_runs_from_its_oldest_format_to_the_current_one():
    formats = [row.reads for row in STATE_MIGRATIONS]
    assert formats == list(range(formats[0], STATE_FORMAT_VERSION))
    assert len({row.name for row in STATE_MIGRATIONS}) == len(STATE_MIGRATIONS)


@pytest.mark.parametrize("version", [0, 2])
def test_a_format_older_than_the_table_is_refused(tmp_path: Path, version: int):
    # no workspace or folder: the format is refused before either is read
    migration = FolderMigration(workspace=None, layout=None)
    with pytest.raises(ValueError, match="older pre-release LangSlice"):
        current_state({"format_version": version, "slices": []}, migration)
    assert not migration.used


# --- what LangSlice no longer has, dropped with a warning ----------------------------------


def _old_folder(tmp_path: Path) -> Path:
    """A job folder as an older LangSlice left it: removed settings in
    ``job.json``, removed keys in ``state.json`` and in an undo step."""
    images = tmp_path / "stack"
    images.mkdir()
    for index in range(2):
        Image.fromarray(np.full((30, 40, 3), 60 + 20 * index, dtype=np.uint8)).save(
            images / f"s{index}.png")
    spec = JobSpec(image_folder=str(images), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    with job.writing():
        before = job.snapshot()
        job.state.slices[0].position_mm = 5.0
        job.commit(before)
    job.close()
    folder = images / "langslice"

    record = json.loads((folder / "job.json").read_text())
    record["spec"]["max_input_tokens"] = None
    record["spec"]["transform"]["flip"] = True
    record["spec"]["tasks"] = ["reorder", *record["spec"]["tasks"]]
    (folder / "job.json").write_text(json.dumps(record))
    old_rows = {"damaged": True, "position_source": ""}
    for path in [folder / "state.json", *sorted((folder / "history").glob("step-*.json"))]:
        state = json.loads(path.read_text())
        state["appearance"] = {}
        state["slices"] = [{**row, **old_rows} for row in state["slices"]]
        path.write_text(json.dumps(state))
    return folder


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == "langslice.job.migrations" and record.levelname == "WARNING"]


def test_an_old_folder_opens_with_one_warning_per_removed_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    folder = _old_folder(tmp_path)
    with caplog.at_level(logging.WARNING):
        handle = langslice.open_job(folder, atlas_loader=lambda _n: _ATLAS)
    handle.close()
    messages = _warnings(caplog)
    named = ["max_input_tokens", "transform.flip", "tasks: reorder", "'appearance'",
             "'damaged' (in 2 section(s))", "'position_source' (in 2 section(s))"]
    assert len(messages) == len(named)
    for name in named:
        assert sum(name in message for message in messages) == 1, name

    # written once without them; the events say what was dropped
    spec = json.loads((folder / "job.json").read_text())["spec"]
    assert "max_input_tokens" not in spec and "flip" not in spec["transform"]
    assert "reorder" not in spec["tasks"]
    for path in [folder / "state.json", *sorted((folder / "history").glob("step-*.json"))]:
        state = json.loads(path.read_text())
        assert "appearance" not in state
        assert all("damaged" not in row and "position_source" not in row
                   for row in state["slices"])
    events = [json.loads(line) for line in
              (folder / "logs" / "events.jsonl").read_text().splitlines()]
    migrated = [event for event in events if event["kind"] == "migrated"]
    assert len(migrated) == 1
    assert migrated[0]["settings"] == ["max_input_tokens", "transform.flip", "tasks: reorder"]
    assert migrated[0]["dropped"] == {"appearance": 1, "slices.position_source": 2,
                                      "slices.damaged": 2}

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        langslice.open_job(folder, atlas_loader=lambda _n: _ATLAS).close()
    assert _warnings(caplog) == []


def test_a_dry_run_of_an_old_folder_warns_and_writes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    from langslice.doors.jobs import open_folder

    folder = _old_folder(tmp_path)
    before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    with caplog.at_level(logging.WARNING):
        opened = open_folder(folder, atlas_loader=lambda _n: _ATLAS, persist=False)
    assert opened.job.state.slices[0].position_mm == 5.0
    opened.close()
    assert len(_warnings(caplog)) == 6
    after = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    assert after == before
