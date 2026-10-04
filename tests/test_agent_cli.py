"""The agent CLI: ``langslice ops``, ``schema``, ``job FOLDER VERB``.

Run through ``langslice.cli.main`` as an agent runs it: one JSON envelope on
stdout (``ok``, ``result``, ``artifacts``, ``warnings``, ``next``, and
``error`` with a ``fix`` when not ok), exit codes 0 / 2 (bad arguments) /
3 (refused by the job) / 4 (internal). The stack is the golden recorder's
synthetic one (``tests/golden/record.py``), its atlas installed where every
CLI job opens (``tests.cli_child.install``); background runs start
``python -m tests.cli_child`` in a process of their own.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from langslice.cli import main
from tests.cli_child import install
from tests.golden.record import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    apply_patches,
    atlas_loader,
    write_sections,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("cli") / "stack"
    write_sections(images)
    return images


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh copy of the stack, the synthetic atlas installed, HOME private."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    install(atlas_loader(), monkeypatch.setattr)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    return folder


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    """Run one command; ``(exit code, envelope)``; stdout must be the envelope alone."""
    code = main(list(argv))
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1, f"stdout must hold one JSON line, got: {out!r}"
    return int(code or 0), json.loads(lines[0])


def init(capsys: pytest.CaptureFixture[str], images: Path, *flags: str) -> Path:
    code, envelope = cli(capsys, "job", str(images), "init", "--tasks",
                         "reorder,position,transform,nonlinear", "--image-provider", "none",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         *flags)
    assert code == 0, envelope
    return Path(envelope["result"]["job_folder"])


# --- ops and schema -----------------------------------------------------------------


def test_ops_lists_every_verb_with_kind_group_and_one_line(capsys):
    from langslice.ops.registry import VERBS

    code, envelope = cli(capsys, "ops")
    assert code == 0 and envelope["ok"] is True
    listed = envelope["result"]["verbs"]
    assert [row["name"] for row in listed] == list(VERBS)
    for row in listed:
        assert row["kind"] == VERBS[row["name"]].kind
        assert row["group"] == VERBS[row["name"]].group
        assert row["summary"] and "\n" not in row["summary"]
    assert set(envelope) == {"ok", "result", "artifacts", "warnings", "next"}


def test_schema_of_one_verb_and_of_every_verb(capsys):
    code, envelope = cli(capsys, "schema", "set-positions")
    assert code == 0
    result = envelope["result"]
    assert result["verb"] == "set_positions" and result["schema_version"] == 1
    schema = result["arguments"]
    assert schema["required"] == ["entries"]
    assert "resolution" in schema["$defs"]["ViewAuto"]["properties"]
    code, everything = cli(capsys, "schema")
    assert code == 0 and "fit_deformable" in everything["result"]["verbs"]
    code, unknown = cli(capsys, "schema", "align_everything")
    assert code == 2 and unknown["error"]["code"] == "UNKNOWN_VERB"
    assert "langslice ops" in unknown["error"]["fix"]


# --- init, the card, status ---------------------------------------------------------


def test_init_creates_the_job_and_its_reference_card(capsys, images):
    job = init(capsys, images)
    assert job == images / "langslice"
    agents = (job / "AGENTS.md").read_text()
    assert agents == (job / "CLAUDE.md").read_text()
    for needle in ("state.json", "coordinate_map", "langslice job", "set_positions",
                   "fit_deformable", "langslice.open_job", "i * resolution"):
        assert needle in agents
    assert len(agents.splitlines()) < 75
    code, envelope = cli(capsys, "job", str(images), "status")
    assert code == 0
    assert [row["id"] for row in envelope["result"]["rows"]] == [ID0, ID1, ID2]
    assert "fit_deformable" in envelope["result"]["verbs"]
    # Opening the job folder itself works too; a folder without a job is refused.
    code, _envelope = cli(capsys, "job", str(job), "status")
    assert code == 0
    code, missing = cli(capsys, "job", str(images.parent), "status")
    assert code == 2 and missing["error"]["code"] == "NO_JOB"
    # A stale card is rewritten on the next open.
    (job / "AGENTS.md").write_text("old")
    cli(capsys, "job", str(images), "status")
    assert (job / "AGENTS.md").read_text() == agents


# --- writes, pictures, exit codes -----------------------------------------------------


def test_a_write_returns_its_pictures_as_artifact_paths(capsys, images):
    init(capsys, images)
    code, envelope = cli(capsys, "job", str(images), "set-positions", "--args",
                         json.dumps({"entries": [{"id": ID0, "position_mm": 0.1}]}),
                         "--view", '{"mode": "overlay", "resolution": 300}')
    assert code == 0, envelope
    assert envelope["result"]["written"] == [{"id": ID0, "position_mm": 0.1}]
    assert "description" not in envelope["result"] and "images" not in envelope["result"]
    kinds = [artifact["kind"] for artifact in envelope["artifacts"]]
    assert kinds == ["view", "view_json", "labels", "borders"]
    for artifact in envelope["artifacts"]:
        assert Path(artifact["path"]).is_file() and os.path.isabs(artifact["path"])
    record = json.loads(Path(envelope["artifacts"][1]["path"]).read_text())
    assert record["tool"] == "set_positions" and record["sections"] == [ID0]
    from PIL import Image

    with Image.open(envelope["artifacts"][0]["path"]) as picture:
        assert max(picture.size) <= 300 + 64  # the caller's size (the caption band on top)
    # The flag form, a corrected index as a number, the verbose reply.
    code, verbose = cli(capsys, "job", str(images), "set_positions", "--entries",
                        '[{"id": 1, "position_mm": 0.15}]', "--verbose")
    assert code == 0 and "description" in verbose["result"]


def test_exit_codes_and_fix_hints(capsys, images):
    init(capsys, images)
    code, envelope = cli(capsys, "job", str(images), "set_positions", "--entries", "[]",
                         "--bogus", "1")
    assert code == 2 and envelope["ok"] is False
    assert envelope["error"]["code"] == "UNKNOWN_ARGUMENTS"
    assert "langslice schema set_positions" in envelope["error"]["fix"]
    code, envelope = cli(capsys, "job", str(images), "view_slices")
    assert code == 2 and envelope["error"]["code"] == "MISSING_ARGUMENTS"
    code, envelope = cli(capsys, "job", str(images), "view_slices", "--slices", "nope.png")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_SLICE_IDS"
    code, envelope = cli(capsys, "job", str(images), "set_positions", "--args", "{not json")
    assert code == 2 and envelope["error"]["code"] == "BAD_JSON"
    code, envelope = cli(capsys, "job", str(images), "undo")
    assert code == 3 and envelope["error"]["code"] == "NOTHING_TO_UNDO"
    code, envelope = cli(capsys, "job", str(images), "submit", "--summary", "done",
                         "--notes", "[]", "--interval-breaks", "[]")
    assert code == 3 and envelope["error"]["code"] == "MISSING_POSITIONS"
    assert envelope["error"]["fix"] and envelope["result"]["missing_ids"] == [ID0, ID1, ID2]
    code, envelope = cli(capsys, "job", str(images), "search_position", "--id", ID0)
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"


def test_an_internal_error_exits_4(capsys, images, monkeypatch):
    init(capsys, images)
    import langslice.ops.notes as notes

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(notes, "add_note", broken)
    code, envelope = cli(capsys, "job", str(images), "note", "--text", "hello")
    assert code == 4 and envelope["error"]["code"] == "INTERNAL"
    assert "boom" in envelope["error"]["message"]


def test_dry_run_reports_the_change_and_writes_nothing(capsys, images):
    job = init(capsys, images)
    state_before = (job / "state.json").read_bytes()
    views_before = sorted(path.name for path in job.rglob("*"))
    code, envelope = cli(capsys, "job", str(images), "set_positions", "--entries",
                         json.dumps([{"id": ID1, "position_mm": 0.15}]), "--dry-run")
    assert code == 0, envelope
    assert envelope["result"]["dry_run"] is True
    assert envelope["result"]["would_change"] == {"sections": {ID1: ["position_mm"]},
                                                  "stack": []}
    assert envelope["artifacts"] == []
    assert envelope["next"] and "--dry-run" not in envelope["next"][0]
    assert (job / "state.json").read_bytes() == state_before
    assert sorted(path.name for path in job.rglob("*")) == views_before
    # A fit is checked, not run.
    code, envelope = cli(capsys, "job", str(images), "fit_deformable", "--slices", ID0,
                         "--dry-run")
    assert code == 0 and envelope["result"] == {"dry_run": True, "simulated": False,
                                                "sections": [ID0]}


def test_the_cli_never_applies_the_look_before_commit_gates(capsys, images):
    init(capsys, images, "--gates")
    for name, position in ((ID0, 0.1), (ID1, 0.15), (ID2, 0.2)):
        code, envelope = cli(capsys, "job", str(images), "set_positions", "--entries",
                             json.dumps([{"id": name, "position_mm": position}]))
        assert code == 0 and envelope["result"]["rejected"] == [], envelope
    code, envelope = cli(capsys, "job", str(images), "submit", "--summary", "done",
                         "--notes", "[]", "--interval-breaks", "[]")
    # Past the positions: the transforms gate (the job's rule), not NOT_REVIEWED.
    assert code == 3 and envelope["error"]["code"] == "MISSING_TRANSFORMS"


# --- background runs --------------------------------------------------------------------


def test_a_background_run_answers_at_once_then_status_and_wait(capsys, images, monkeypatch):
    from langslice.doors.cli import background

    init(capsys, images)
    # install() pointed background runs at tests.cli_child (the synthetic atlas).
    assert background.CHILD_COMMAND[-1] == "tests.cli_child"
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.chdir(REPO)
    code, envelope = cli(capsys, "job", str(images), "view_slices", "--slices", ID0,
                         "--background")
    assert code == 0 and envelope["result"]["state"] == "running"
    run = envelope["result"]["run"]
    assert envelope["next"] == [f"langslice job {images / 'langslice'} wait {run}"]
    code, waited = cli(capsys, "job", str(images), "wait", run, "--timeout", "120")
    assert code == 0, waited
    assert waited["result"]["run"] == run and waited["result"]["slices"] == [ID0]
    assert [artifact["kind"] for artifact in waited["artifacts"]] == ["view", "view_json"]
    assert Path(waited["artifacts"][0]["path"]).is_file()
    code, one = cli(capsys, "job", str(images), "runs", run)
    assert code == 0 and one["result"]["state"] == "finished"
    assert one["result"]["exit"] == 0
    code, listed = cli(capsys, "job", str(images), "runs")
    assert listed["result"]["runs"][0]["id"] == run
    code, unknown = cli(capsys, "job", str(images), "runs", "no-such-run")
    assert code == 2 and unknown["error"]["code"] == "UNKNOWN_RUN"
    # `status` is only the verb.
    code, status = cli(capsys, "job", str(images), "status", run)
    assert code == 2 and status["error"]["code"] == "BAD_ARGUMENTS"


# --- live shared editing --------------------------------------------------------------


def test_cli_calls_and_a_running_toolbox_interleave_on_one_folder(capsys, images):
    from langslice.doors.jobs import context
    from langslice.linear.job import Job
    from langslice.linear.spec import JobSpec, NonlinearSpec, PositionSpec
    from langslice.linear.toolbox import build_tools

    job_folder = init(capsys, images)
    # The agent's side: the job folder opened as a run opens it, its tools.
    spec = JobSpec(image_folder=str(images), preprocess="none", resume=True,
                   tasks=["reorder", "position", "transform", "nonlinear"],
                   inputs={"pixel_size_um": PIXEL_SIZE_UM}, position=PositionSpec(),
                   nonlinear=NonlinearSpec(provider="none"))
    ctx = context(spec, job_folder)
    agent = Job.open(spec, ctx, folder=job_folder, results_path=ctx.results_path)
    tools = {tool.__name__: tool for tool in build_tools(agent.state, ctx, spec,
                                                         job=agent).tools}
    assert tools["set_positions"]([{"id": ID0, "position_mm": 0.1}])["status"] == "ok"

    # The CLI sees the agent's write and adds its own.
    code, envelope = cli(capsys, "job", str(images), "set_positions", "--entries",
                         json.dumps([{"id": ID1, "position_mm": 0.15}]))
    assert code == 0
    code, envelope = cli(capsys, "job", str(images), "status")
    positions = {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]}
    assert positions == {ID0: 0.1, ID1: 0.15, ID2: None}

    # The agent's next call picks up the CLI's write, history included.
    rows = tools["status"]()["rows"]
    assert {row["id"]: row.get("position_mm") for row in rows} == positions
    assert tools["undo"]()["status"] == "ok"  # undoes the CLI's write
    assert agent.state.resolve(ID1).position_mm is None
    assert agent.state.resolve(ID0).position_mm == 0.1

    # And the CLI sees the agent's undo; its own undo takes the agent's write back.
    code, envelope = cli(capsys, "job", str(images), "undo")
    assert code == 0
    code, envelope = cli(capsys, "job", str(images), "status")
    assert all(row.get("position_mm") is None for row in envelope["result"]["rows"])
    assert tools["status"]()["rows"][0].get("position_mm") is None
    agent.close()


# --- the library ------------------------------------------------------------------------


def test_open_job_gives_the_verbs_as_methods(capsys, images):
    import langslice

    init(capsys, images)
    with langslice.open_job(images) as job:
        assert "set_positions" in job.verbs and "set_positions" in dir(job)
        reply = job.set_positions(entries=[{"id": ID2, "position_mm": 0.2}],
                                  view={"mode": "overlay"})
        assert reply["status"] == "ok" and reply["images"]
        assert job.state.resolve(ID2).position_mm == 0.2
        with pytest.raises(AttributeError, match="search_position"):
            getattr(job, "search_position")  # noqa: B009 — a missing verb
    code, envelope = cli(capsys, "job", str(images), "status")
    assert envelope["result"]["rows"][2]["position_mm"] == 0.2
    saved = next((images / "langslice" / "sections").rglob("view.json"))
    xyz = langslice.coordinate_map(saved)
    assert xyz.ndim == 3 and xyz.shape[2] == 3 and bool((xyz == xyz).any())
