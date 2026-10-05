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
    # Every verb but the hidden ones (Verb.hidden: called by name only).
    assert [row["name"] for row in listed] == [
        name for name, verb in VERBS.items() if not verb.hidden]
    assert "trace_from_atlas" not in [row["name"] for row in listed]
    for row in listed:
        assert row["kind"] == VERBS[row["name"]].kind
        assert row["group"] == VERBS[row["name"]].group
        assert row["summary"] and "\n" not in row["summary"]
        assert row.get("long", False) is VERBS[row["name"]].long
    assert "brief" in envelope["result"]["job_commands"]
    assert set(envelope) == {"ok", "result", "artifacts", "warnings", "next"}


def test_schema_of_one_verb_and_of_every_verb(capsys):
    from langslice.doors.declarations import Variant, declaration

    code, envelope = cli(capsys, "schema", "set-positions")
    assert code == 0
    result = envelope["result"]
    assert result["verb"] == "set_positions" and result["schema_version"] == 2
    schema = result["arguments"]
    assert schema["required"] == ["entries"]
    assert "resolution" in schema["$defs"]["ViewAuto"]["properties"]
    # What a model reads of the verb: the whole description, the summary, the
    # picture options described once (with the resolution range), long or not.
    assert result["description"] == declaration(
        "set_positions", Variant(auto=True, door="cli")).doc
    assert "\n" in result["description"] and result["summary"] in result["description"]
    assert "`view` also takes `resolution`" in result["picture_options"]
    assert "128 to 2000" in result["picture_options"]  # the default viewer's largest
    assert result["long"] is False and result["job"] is None and "--job" in result["hint"]
    code, fit = cli(capsys, "schema", "fit_affine")
    assert fit["result"]["long"] is True and fit["next"][0].endswith("--background")
    code, everything = cli(capsys, "schema")
    assert code == 0 and "fit_deformable" in everything["result"]["verbs"]
    assert everything["result"]["verbs"]["status"]["arguments"]["properties"] == {}
    assert "picture_options" not in everything["result"]["verbs"]["status"]
    # A hidden verb: not in the listing, its schema by name.
    assert "trace_from_atlas" not in everything["result"]["verbs"]
    code, hidden = cli(capsys, "schema", "trace-from-atlas")
    assert code == 0 and hidden["result"]["arguments"]["required"] == ["slices"]
    code, unknown = cli(capsys, "schema", "align_everything")
    assert code == 2 and unknown["error"]["code"] == "UNKNOWN_VERB"
    assert "langslice ops" in unknown["error"]["fix"]


def test_schema_declares_a_jobs_own_verbs(capsys, images, monkeypatch):
    init(capsys, images, "--viewer", "codex", "--engine", "elastix")
    code, envelope = cli(capsys, "schema", "fit_deformable", "--job", str(images))
    assert code == 0, envelope
    result = envelope["result"]
    assert result["job"] == str(images / "langslice") and "hint" not in result
    assert "engine" not in result["arguments"]["properties"]  # the user fixed it
    assert "128 to 2048" in result["picture_options"]  # the job's viewer
    assert "Raw image channels per section" in result["picture_options"]
    # Run in a job folder, schema declares that job's verbs.
    monkeypatch.chdir(images / "langslice")
    code, here = cli(capsys, "schema", "fit_deformable")
    assert here["result"]["job"] == str(images / "langslice")
    assert "engine" not in here["result"]["arguments"]["properties"]


def test_schema_of_an_unreadable_job_is_one_envelope(capsys, images):
    job = init(capsys, images)
    record = json.loads((job / "job.json").read_text())
    record["spec"]["plane"] = "diagonal"  # a spec the job layer refuses
    (job / "job.json").write_text(json.dumps(record))
    code, envelope = cli(capsys, "schema", "fit_deformable", "--job", str(images))
    assert code == 3 and envelope["error"]["code"] == "JOB_UNREADABLE"


# --- init, the card, status ---------------------------------------------------------


def test_init_creates_the_job_and_its_reference_card(capsys, images):
    job = init(capsys, images)
    assert job == images / "langslice"
    agents = (job / "AGENTS.md").read_text()
    assert agents == (job / "CLAUDE.md").read_text()
    for needle in ("state.json", "coordinate_map", "langslice job", "set_positions",
                   "fit_deformable", "langslice.open_job", "i * resolution", " brief`",
                   "BRIEF.md", "STALE_INPUT", "(write, Linear, long)"):
        assert needle in agents
    assert len(agents.splitlines()) < 85  # one screen, its verb list included
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


def test_cli_json_preserves_unicode_on_non_utf8_consoles():
    from langslice.doors.cli.envelope import Envelope, dumps

    message = "The atlas border — preserved exactly"
    wire = dumps(Envelope(result={"statement": message}))
    assert wire.isascii()
    assert json.loads(wire)["result"]["statement"] == message


# --- brief: the job statement and the opening, as LangSlice's own agent gets them ------


def _native(images: Path, limit: tuple[int, int]) -> tuple[str, list[bytes], Any]:
    """The ADK agent's statement and opening picture bytes for the job in *images*."""
    from langslice.agent.prompt import build_job_statement, display_facts
    from langslice.doors.jobs import open_folder
    from langslice.doors.tools.media import opening_parts
    from langslice.doors.tools.toolbox import build_tools

    opened = open_folder(images, atlas_loader=atlas_loader(), persist=False)
    job, ctx = opened.job, opened.ctx
    box = build_tools(job.state, ctx, job.spec, job=job, max_view_edge=limit[0])
    low, high = ctx.position_range
    statement = build_job_statement(
        job.spec, job.state, tool_names=box.names, species=ctx.species, pos_lo=low,
        pos_hi=high, axis_ends=ctx.axis_ends, max_resolution=limit[0], auto=True,
        **display_facts(ctx, job.state))
    parts = opening_parts(job.state, ctx, limit=limit)
    pictures = [part.inline_data.data for part in parts if part.inline_data is not None]
    return statement, pictures, opened


def test_brief_is_the_native_statement_and_opening(capsys, images):
    from langslice.core.opening import DEFAULT_IMAGE_LIMIT

    job = init(capsys, images, "--notes", "Section 2 is torn.", "--viewer", "codex")
    code, envelope = cli(capsys, "job", str(images), "brief")
    assert code == 0, envelope
    result = envelope["result"]
    statement = result["statement"]
    native, pictures, opened = _native(images, DEFAULT_IMAGE_LIMIT)
    try:
        # The same text, but for where this door's opening pictures are.
        expected = native.replace("in the opening message)", "in the opening pictures (brief))")
        assert expected != native and statement.startswith(expected + "\n\n")
        tail = statement[len(expected):]
        assert "Opening pictures: `brief` saved 2 picture files" in tail
        assert "User notes:\nSection 2 is torn." in tail
        assert "trace_from_atlas" not in statement  # a hidden verb is listed nowhere
        from langslice.doors.statement import status_and_notes

        assert tail.endswith(status_and_notes(opened.job.state))
    finally:
        opened.close()
    # The opening pictures: the native strips, byte for byte, in reading order.
    opening = [item for item in envelope["artifacts"] if item["kind"] == "opening"]
    assert [item["index"] for item in opening] == list(range(len(pictures)))
    assert [Path(item["path"]).read_bytes() for item in opening] == pictures
    assert opening[0]["label"].startswith("Strip 1 of 1: 0: s0.png")
    assert [entry.get("path") for entry in result["opening"] if "picture" in entry] == [
        item["path"] for item in opening]
    # BRIEF.md holds it all, and the card names it first.
    brief = (job / "BRIEF.md").read_text()
    normalized = statement.replace("\r\n", "\n")
    saved = brief.split("## Job statement\n\n", 1)[1]
    mismatch = next((i for i, (actual, expected) in enumerate(zip(normalized, saved, strict=False))
                     if actual != expected), min(len(normalized), len(saved)))
    assert normalized in brief, (f"first mismatch at {mismatch}: "
                                 f"{normalized[max(0, mismatch - 30):mismatch + 80]!r} vs "
                                 f"{saved[max(0, mismatch - 30):mismatch + 80]!r}")
    assert all(item["path"] in brief for item in opening)
    assert envelope["artifacts"][-1] == {"path": str(job / "BRIEF.md"), "kind": "brief"}
    assert "BRIEF.md" in (job / "AGENTS.md").read_text()
    assert result["viewer"] == "codex" and result["resolution"]["max"] == 2048


def test_brief_writes_normalized_line_endings_on_every_platform(tmp_path: Path):
    from langslice.doors.cli.brief import Brief, write

    saved = write(tmp_path, Brief(statement="First line\r\nSecond line"), pictures=False)
    assert saved is not None
    data = saved.read_bytes()
    assert b"First line\nSecond line" in data
    assert b"\r" not in data


def test_init_answers_with_the_statement_and_keeps_notes_and_viewer(capsys, images):
    code, envelope = cli(capsys, "job", str(images), "init", "--tasks", "position",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         "--notes", "Mind the bulbs.")
    assert code == 0, envelope
    statement = envelope["result"]["statement"]
    assert "You are an expert neuroanatomist" in statement
    assert "`langslice job " in statement and " brief` saves them as picture files" in statement
    assert "User notes:\nMind the bulbs." in statement
    assert envelope["result"]["viewer"] == "claude"  # the default viewer
    assert envelope["next"] == [f"langslice job {images / 'langslice'} brief"]
    record = json.loads((images / "langslice" / "job.json").read_text())
    assert record["notes"] == "Mind the bulbs." and "viewer" not in record
    # Continuing the job keeps the notes; --viewer is stored.
    code, again = cli(capsys, "job", str(images), "init", "--tasks", "position",
                      "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                      "--viewer", "openai")
    assert code == 0 and again["result"]["viewer"] == "openai"
    record = json.loads((images / "langslice" / "job.json").read_text())
    assert record["notes"] == "Mind the bulbs." and record["viewer"] == "openai"


def test_pictures_are_capped_at_the_viewers_largest(capsys, images):
    init(capsys, images)  # viewer claude (default): up to 2000 px
    code, envelope = cli(capsys, "job", str(images), "view-slices", "--slices", ID1,
                         "--view", '{"resolution": 5000}')
    assert code == 0, envelope
    assert "used 2000" in envelope["result"]["view"]["resolution_note"]
    init(capsys, images, "--viewer", "codex")
    code, envelope = cli(capsys, "job", str(images), "view-slices", "--slices", ID1,
                         "--view", '{"resolution": 5000}')
    assert "used 2048" in envelope["result"]["view"]["resolution_note"]


def test_an_image_model_not_connected_is_reported_and_its_verb_refused(
        capsys, images, monkeypatch):
    init(capsys, images, "--image-provider", "openai-oauth")  # HOME is private: no login
    code, status = cli(capsys, "job", str(images), "status")
    assert code == 0
    assert status["result"]["image_model"] == {
        "provider": "openai-oauth", "connected": False, "trace_borders": False}
    assert "trace_borders" not in status["result"]["verbs"]
    code, refused = cli(capsys, "job", str(images), "trace_borders", "--id", ID0)
    assert code == 3 and refused["error"]["code"] == "IMAGE_MODEL_OFF"
    assert "langslice login" in refused["error"]["fix"]
    code, brief = cli(capsys, "job", str(images), "brief")
    from langslice.doors.statement import IMAGE_MODEL_OFF

    assert IMAGE_MODEL_OFF in brief["result"]["statement"]
    # Connected (a login present), the verb is offered, as the MCP door offers it.
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "image_model_connected", lambda provider: True)
    code, status = cli(capsys, "job", str(images), "status")
    assert status["result"]["image_model"]["connected"] is True
    assert "trace_borders" in status["result"]["verbs"]


def test_status_marks_what_the_user_locked_or_marked_damaged(capsys, images):
    init(capsys, images, "--locked", json.dumps([ID0]), "--damaged",
         json.dumps({ID1: "torn"}))
    code, envelope = cli(capsys, "job", str(images), "status")
    assert code == 0
    rows = {row["id"]: row for row in envelope["result"]["rows"]}
    assert rows[ID0].get("locked") is True and "damage_by_user" not in rows[ID0]
    assert rows[ID1].get("damage_by_user") is True and "locked" not in rows[ID1]
    assert "locked" not in rows[ID2] and "damage_by_user" not in rows[ID2]


def test_every_call_is_logged_and_traced(capsys, images, monkeypatch, tmp_path):
    traces = tmp_path / "traces"
    monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(traces))
    job = init(capsys, images)
    code, envelope = cli(capsys, "job", str(images), "view_slices", "--slices", ID0)
    assert code == 0
    code, refused = cli(capsys, "job", str(images), "undo")
    assert code == 3
    lines = [json.loads(line) for line in (job / "logs" / "calls.jsonl").read_text()
             .splitlines()]
    assert [line["verb"] for line in lines] == ["init", "view_slices", "undo"]
    seen = lines[1]
    assert seen["arguments"] == ["--slices", ID0] and seen["ok"] is True
    assert seen["exit"] == 0 and seen["artifacts"] == [
        item["path"] for item in envelope["artifacts"]]
    assert lines[2]["ok"] is False and lines[2]["error"] == "NOTHING_TO_UNDO"
    # The trace, as the MCP door writes its own: one file for the job.
    files = list(traces.glob("cli_stack_*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [record["kind"] for record in records] == ["tool_result"] * 3
    traced = records[1]
    assert traced["name"] == "view_slices" and traced["args"]["slices"] == [ID0]
    assert json.loads(traced["content"][0]["text"])["ok"] is True
    assert traced["content"][1]["image"] == "image/jpeg" and traced["content"][1]["bytes"] > 0
    # The artifacts: index and label for each picture, as image_indexes count.
    view = envelope["artifacts"][0]
    assert view["kind"] == "view" and view["index"] == 0 and ID0 in view["label"]
    assert "picture_note" in envelope["result"] and "\n" not in envelope["result"][
        "picture_note"]


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
    # A dry run never starts a background run (that child would run the verb).
    code, envelope = cli(capsys, "job", str(images), "fit_deformable", "--slices", ID0,
                         "--dry-run", "--background")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS"
    assert not (job / "logs" / "runs").exists()
    assert (job / "state.json").read_bytes() == state_before


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


def test_export_maps_writes_the_derived_files_as_artifacts(capsys, images):
    job = init(capsys, images)
    code, envelope = cli(capsys, "job", str(images), "export_maps")
    # Nothing placed yet: every section skipped, with its reason.
    assert code == 3 and envelope["error"]["code"] == "NOTHING_EXPORTED"
    assert {row["reason"] for row in envelope["result"]["skipped"]} == {"no position"}
    cli(capsys, "job", str(images), "set_positions", "--entries", json.dumps(
        [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
         {"id": ID2, "position_mm": 0.2}]))
    code, envelope = cli(capsys, "job", str(images), "export-maps", "--slices", ID0,
                         "--dry-run")
    assert code == 0 and envelope["result"]["files_written"] is False
    assert envelope["artifacts"] == [] and not (job / "sections/s0/coords.tif").exists()
    code, envelope = cli(capsys, "job", str(images), "export_maps", "--slices", ID0)
    assert code == 0, envelope
    assert envelope["result"]["written"] == [ID0] and envelope["result"]["n_files"] >= 5
    kinds = {artifact["kind"] for artifact in envelope["artifacts"]}
    assert {"coords", "labels", "labels_fiji", "labels_csv", "maps", "quicknii",
            "visualign", "registration"} <= kinds
    assert all(Path(artifact["path"]).is_file() for artifact in envelope["artifacts"])
    # The verb is the CLI's and the library's: listed by status, never a model's tool.
    code, envelope = cli(capsys, "job", str(images), "status")
    assert "export_maps" in envelope["result"]["verbs"]


# --- background runs --------------------------------------------------------------------


def test_a_background_run_answers_at_once_then_status_and_wait(capsys, images, monkeypatch):
    from langslice.doors.cli import background

    init(capsys, images)
    # install() pointed background runs at tests.cli_child (the synthetic atlas).
    assert background.CHILD_COMMAND[-1] == "tests.cli_child"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO / "src"), str(REPO))))
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
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec
    from langslice.doors.jobs import context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import Job

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


# --- a dead background run on Windows ------------------


def _dead_run(tmp_path: Path) -> tuple[Any, str]:
    from langslice.doors.cli import background
    from langslice.job.layout import JobLayout

    layout = JobLayout(tmp_path / "job")
    background._write(layout, {"id": "run-1", "verb": "fit_deformable", "state": "running",
                               "started": 0.0, "pid": 424242})
    return layout, "run-1"


@pytest.mark.parametrize("probe", ["psutil", "ctypes"])
def test_a_dead_run_reads_lost_on_windows(tmp_path, monkeypatch, probe):
    """On Windows the run's process is really probed (psutil when installed,
    else OpenProcess/GetExitCodeProcess), and ``wait`` without a timeout
    returns once the process is gone."""
    import sys
    import threading
    import types

    from langslice.doors.cli import background

    monkeypatch.setattr(background, "WINDOWS", True)
    if probe == "psutil":
        fake = types.SimpleNamespace(pid_exists=lambda pid: pid != 424242)
        monkeypatch.setitem(sys.modules, "psutil", fake)
    else:
        monkeypatch.setitem(sys.modules, "psutil", None)  # not installed

        class Kernel32:
            def OpenProcess(self, access, inherit, pid):  # noqa: N802
                return 0 if pid == 424242 else 7

            def GetLastError(self):  # noqa: N802
                return 87  # ERROR_INVALID_PARAMETER: no such process

            def GetExitCodeProcess(self, handle, code):  # noqa: N802
                code._obj.value = 259
                return 1

            def CloseHandle(self, handle):  # noqa: N802
                return 1

        monkeypatch.setattr(background, "_kernel32", lambda: Kernel32())
    assert background._alive(4242) is True
    assert background._alive(424242) is False
    layout, run_id = _dead_run(tmp_path)
    assert background.read(layout, run_id)["state"] == "lost"
    answer: list[Any] = []
    waiter = threading.Thread(target=lambda: answer.append(background.wait(layout, run_id, None)),
                              daemon=True)
    waiter.start()
    waiter.join(5.0)
    assert answer and answer[0]["state"] == "lost"
