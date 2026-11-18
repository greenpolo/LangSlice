"""The agent CLI: ``langslice-job ops``, ``schema``, ``job FOLDER VERB``.

Run through ``langslice.cli.main`` as an agent runs it: one JSON envelope on
stdout (``ok``, ``result``, ``artifacts``, ``warnings``, ``next``, and
``error`` with a ``fix`` when not ok), exit codes 0 / 2 (bad arguments) /
3 (refused by the job) / 4 (internal). The stack is the shared
synthetic one (``tests/synthetic_stack.py``), its atlas installed where every
CLI job opens (``tests.cli_child.install``); background runs start
``python -m tests.cli_child`` in a process of their own.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langslice.doors.agent_cli.main import main
from tests.cli_child import install
from tests.synthetic_stack import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
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


def place(images: Path, *names: str) -> None:
    """Place the named sections (all three by default) through the library."""
    import langslice

    positions = {ID0: 0.1, ID1: 0.15, ID2: 0.2}
    with langslice.open_job(images) as job:
        reply = job.position_sections(
            sections=[{"id": name, "position_mm": positions[name]}
                      for name in names or positions], view=False)
    assert reply["status"] == "ok", reply


def init(capsys: pytest.CaptureFixture[str], images: Path, *flags: str) -> Path:
    code, envelope = cli(capsys, str(images), "init", "--tasks",
                         "position,transform,nonlinear", "--nonlinear-provider", "none",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         *flags)
    assert code == 0, envelope
    return Path(envelope["result"]["job_folder"])



def test_schema_of_one_verb_and_of_every_verb(capsys):
    from langslice.doors.agent_cli.catalog import SCHEMA_VERSION
    from langslice.tools.descriptions.description import DescriptionFacts
    from langslice.tools.descriptions.render import FULL, declaration

    assert SCHEMA_VERSION == 3
    code, envelope = cli(capsys, "schema", "interactive-transform")
    assert code == 0
    result = envelope["result"]
    assert result["verb"] == "interactive_transform" and result["schema_version"] == 3
    schema = result["arguments"]
    assert schema["required"] == ["sections"]
    assert {"id", "flip", "rotate_quarter", "rotation_deg", "scale_x", "scale_y", "shear",
            "translate_x_mm", "translate_y_mm"} == set(
        schema["$defs"]["SectionTransform"]["properties"])
    assert result["description"] == declaration(
        "interactive_transform",
        DescriptionFacts(auto=True, door="cli", uncapped=True, prompt=FULL.prompt)).doc
    assert "\n" in result["description"] and result["summary"] in result["description"]
    assert "picture_options" not in result
    assert result["long"] is False and result["job"] is None and "--job" in result["hint"]
    code, look = cli(capsys, "schema", "look")
    assert code == 0 and "resolution" in look["result"]["arguments"]["properties"]
    code, fit = cli(capsys, "schema", "elastix_affine")
    assert fit["result"]["long"] is True and not fit["next"][0].endswith("--background")
    code, everything = cli(capsys, "schema")
    assert code == 0 and "fireants_syn" in everything["result"]["verbs"]
    assert everything["result"]["schema_version"] == 3
    assert everything["result"]["verbs"]["status"]["arguments"]["properties"] == {}
    assert not any("picture_options" in entry
                   for entry in everything["result"]["verbs"].values())
    code, unknown = cli(capsys, "schema", "align_everything")
    assert code == 2 and unknown["error"]["code"] == "UNKNOWN_VERB"
    assert "langslice-job ops" in unknown["error"]["fix"]
    assert "langslice-job FOLDER schema align_everything" in unknown["error"]["message"]


def test_schema_of_several_verbs_at_once(capsys):
    """`schema` takes several verbs (separate or comma-separated) and answers
    with each under `verbs`, job commands included; one unknown refuses all."""
    code, several = cli(capsys, "schema", "look", "zoom,undo", "wait")
    assert code == 0, several
    verbs = several["result"]["verbs"]
    assert list(verbs) == ["look", "zoom", "undo", "wait"]
    assert "resolution" in verbs["look"]["arguments"]["properties"]
    assert "steps" in verbs["undo"]["arguments"]["properties"]
    assert verbs["wait"]["kind"] == "command"
    code, refused = cli(capsys, "schema", "look", "nudge_sections")
    assert code == 2 and refused["error"]["code"] == "UNKNOWN_VERB"
    assert "'nudge_sections'" in refused["error"]["message"]


def test_schema_in_a_jobs_folder_declares_its_verbs(capsys, images):
    """`langslice-job FOLDER schema VERB ...` declares the verbs for that job,
    as `schema --job FOLDER` does (a tool plugin's verbs: test_tool_plugins)."""
    job = init(capsys, images)
    code, both = cli(capsys, str(job), "schema", "look", "fireants_syn")
    assert code == 0, both
    assert both["result"]["job"] == str(job)
    assert set(both["result"]["verbs"]) == {"look", "fireants_syn"}
    code, one = cli(capsys, str(images), "schema", "undo")
    assert code == 0 and one["result"]["verb"] == "undo" and one["result"]["job"] == str(job)


def test_stdout_holds_the_envelope_alone_when_the_atlas_prints(capsys, images, monkeypatch):
    """A library printing on stdout while a job opens (BrainGlobe's version
    notice) lands on stderr: `schema VERB` in a job folder stays one JSON line."""
    init(capsys, images)

    def noisy(name: str) -> Any:
        print("brainglobe_atlasapi: allen_mouse_25um version 3.0 is not the latest")
        return loader(name)

    loader = atlas_loader()
    install(noisy, monkeypatch.setattr)
    code, envelope = cli(capsys, "schema", "status", "--job", str(images))
    assert code == 0 and envelope["result"]["job"] == str(images / "langslice")
    code, envelope = cli(capsys, str(images), "status")
    assert code == 0


def test_load_atlas_keeps_brainglobe_notices_off_stdout(capsys, monkeypatch):
    import brainglobe_atlasapi

    from langslice.core.atlas.load import load_atlas

    class Notice:
        def __init__(self, name: str, **_kwargs: Any) -> None:
            print(f"brainglobe_atlasapi: {name} version 3.0 is not the latest available")
            self.atlas_name = name

    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", Notice)
    atlas = load_atlas.__wrapped__("allen_mouse_25um")
    assert atlas.atlas_name == "allen_mouse_25um"
    captured = capsys.readouterr()
    assert captured.out == "" and "not the latest" in captured.err


def test_schema_declares_a_jobs_own_verbs(capsys, images, monkeypatch):
    code, envelope = cli(capsys, str(images), "init", "--tasks", "transform", "--transform-angles",
                         "--nonlinear-provider", "none", "--preprocess", "none",
                         "--pixel-size-um", str(PIXEL_SIZE_UM), "--viewer", "codex")
    assert code == 0, envelope
    code, everything = cli(capsys, "schema", "position_sections")
    assert "sections" in everything["result"]["arguments"]["properties"]
    code, envelope = cli(capsys, "schema", "position_sections", "--job", str(images))
    assert code == 0, envelope
    result = envelope["result"]
    assert result["job"] == str(images / "langslice") and "hint" not in result
    properties = result["arguments"]["properties"]
    assert "sections" not in properties and "cutting_angles" in properties
    assert "sections: [{" not in result["description"]
    monkeypatch.chdir(images / "langslice")
    code, here = cli(capsys, "schema", "position_sections")
    assert here["result"]["job"] == str(images / "langslice")
    assert "sections" not in here["result"]["arguments"]["properties"]
    code, listed = cli(capsys, "schema", "--job", str(images))
    assert "fireants_syn" not in listed["result"]["verbs"]
    assert "interactive_transform" in listed["result"]["verbs"]
    code, off = cli(capsys, "schema", "fireants_syn", "--job", str(images))
    assert code == 3 and off["error"]["code"] == "VERB_OFF"


def test_schema_of_an_unreadable_job_is_one_envelope(capsys, images):
    job = init(capsys, images)
    record = json.loads((job / "job.json").read_text(encoding="utf-8"))
    record["spec"]["plane"] = "diagonal"
    (job / "job.json").write_text(json.dumps(record), encoding="utf-8")
    code, envelope = cli(capsys, "schema", "fireants_syn", "--job", str(images))
    assert code == 3 and envelope["error"]["code"] == "JOB_UNREADABLE"



def test_init_creates_the_job_and_its_reference_card(capsys, images):
    job = init(capsys, images)
    assert job == images / "langslice"
    agents = (job / "AGENTS.md").read_text(encoding="utf-8")
    assert agents == (job / "CLAUDE.md").read_text(encoding="utf-8")
    from langslice.tools.catalog import VERBS
    from langslice.tools.fitting.deformable import MAX_FIREANTS_SYN_SECTIONS

    for needle in ("state.json", "coordinate_map", "langslice-job", "langslice.open_job",
                   "i * resolution", " launch --role register`", "STALE_INPUT",
                   "- `elastix_affine` (write, Linear, long):",
                   "- `look` (read, Common):",
                   "- `interactive_transform` (write, Linear):",
                   f"- `fireants_syn` (write, Nonlinear, long; at most {MAX_FIREANTS_SYN_SECTIONS} "
                   "sections per call):",
                   'job.position_sections(sections=[{"id": "<file>", "position_mm": 5.2}])',
                   "schema init", "reply.images", "`scripts/`", "never in /tmp",
                   "`env install PACKAGE ...`", "`job.context`"):
        assert needle in agents, needle
    assert len(agents.splitlines()) < 100
    code, envelope = cli(capsys, str(images), "status")
    assert code == 0
    listed = [name for name in VERBS if f"- `{name}` (" in agents]
    assert listed == [name for name in VERBS if name in envelope["result"]["verbs"]]
    assert [row["id"] for row in envelope["result"]["rows"]] == [ID0, ID1, ID2]
    assert {"position_sections", "elastix_affine", "fireants_syn", "export_maps"} <= set(
        envelope["result"]["verbs"])
    code, _envelope = cli(capsys, str(job), "status")
    assert code == 0
    code, missing = cli(capsys, str(images.parent), "status")
    assert code == 2 and missing["error"]["code"] == "NO_JOB"
    (job / "AGENTS.md").write_text("old", encoding="utf-8")
    cli(capsys, str(images), "status")
    assert (job / "AGENTS.md").read_text(encoding="utf-8") == agents



def test_launch_register_scripting_is_the_mcp_statement_with_the_scripts(capsys, images):
    """The scripting agent's message is the MCP agent's with a paragraph on
    scripts after the opening's: the job's own Python (made by the launch),
    how to add a package, the library, where its files go."""
    pytest.importorskip("mcp")
    from langslice.doors.agent_cli import environment
    from langslice.job.folder.layout import JobLayout

    job = init(capsys, images, "--notes", "Section 2 is torn.")
    code, plain = cli(capsys, str(images), "launch", "--role", "register")
    assert code == 0, plain
    assert not (job / "env").exists()
    code, envelope = cli(capsys, str(images), "launch", "--role", "register-scripting")
    assert code == 0, envelope
    result = envelope["result"]
    prompt = result["prompt"]
    assert result["role"] == "register-scripting"
    python = environment.python_of(job / "env")
    assert python.is_file()
    scripts = prompt[prompt.index("\n\nScripts:\n"):prompt.index("\n\nUser notes:")]
    assert prompt.replace(scripts, "") == plain["result"]["prompt"]
    assert f'`"{python}" SCRIPT [ARGS ...]`' in scripts
    assert f"`{environment.install_command(JobLayout(job))} PACKAGE [PACKAGE ...]`" in scripts
    assert f'langslice.open_job(r"{job}")' in scripts and "help(job.context)" in scripts
    assert f"`{job}/scripts/`" in scripts and "langslice-job" not in prompt
    assert "start_job(" in prompt and "show_stack(page=1)" in prompt
    launch = envelope["artifacts"][0]
    assert launch == {"path": str(job / "logs" / "launches" / "000002_register-scripting.md"),
                      "kind": "launch"}
    assert Path(launch["path"]).read_text(encoding="utf-8") == prompt


def test_launch_register_is_the_mcp_statement(capsys, images):
    pytest.importorskip("mcp")
    job = init(capsys, images, "--notes", "Section 2 is torn.")
    code, envelope = cli(capsys, str(images), "launch", "--role", "register")
    assert code == 0, envelope
    prompt = envelope["result"]["prompt"]
    first, _blank, statement = prompt.partition("\n\n")
    assert first == (f"Register the brain in job folder {images}. Call "
                     f'start_job(image_folder="{images}") first: it opens the job and '
                     "gives you its tools.")
    assert "show_stack(page=1)" in statement and "User notes:\nSection 2 is torn." in statement
    saved = job / "logs" / "launches" / "000001_register.md"
    assert saved.read_text(encoding="utf-8") == prompt
    assert envelope["artifacts"] == [{"path": str(saved), "kind": "launch"}]


def test_launch_needs_a_role_and_a_job(capsys, images, tmp_path):
    init(capsys, images)
    code, envelope = cli(capsys, str(images), "launch")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGS"
    code, envelope = cli(capsys, str(images), "launch", "--role", "other")
    assert code == 2
    code, envelope = cli(capsys, str(tmp_path), "launch", "--role", "register")
    assert envelope["error"]["code"] == "NO_JOB"


def test_init_keeps_notes_and_viewer(capsys, images):
    code, envelope = cli(capsys, str(images), "init", "--tasks", "position",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         "--notes", "Mind the bulbs.")
    assert code == 0, envelope
    assert "statement" not in envelope["result"]
    assert envelope["result"]["viewer"] == "claude"
    assert envelope["next"] == [
        f"langslice-job {images / 'langslice'} launch --role register",
        f"langslice-job {images / 'langslice'} launch --role register-scripting"]
    record = json.loads((images / "langslice" / "job.json").read_text(
        encoding="utf-8"))
    assert record["notes"] == "Mind the bulbs." and "viewer" not in record
    code, again = cli(capsys, str(images), "init", "--tasks", "position",
                      "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                      "--viewer", "openai")
    assert code == 0 and again["result"]["viewer"] == "openai"
    record = json.loads((images / "langslice" / "job.json").read_text(
        encoding="utf-8"))
    assert record["notes"] == "Mind the bulbs." and record["viewer"] == "openai"


def test_init_on_an_existing_job_refuses_other_settings(capsys, images):
    """DOORS-2: a second init with other job flags does not replace the saved settings."""
    job = init(capsys, images)
    before = json.loads((job / "job.json").read_text(encoding="utf-8"))["spec"]
    code, envelope = cli(capsys, str(images), "init", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         "--notes", "Mind the bulbs.")
    assert code != 0 and envelope["error"]["code"] == "SETTINGS_CHANGED", envelope
    assert "tasks" in envelope["error"]["message"]
    assert json.loads((job / "job.json").read_text(
        encoding="utf-8"))["spec"] == before


def test_an_image_model_not_connected_is_reported_and_its_verb_refused(
        capsys, images, monkeypatch):
    init(capsys, images, "--nonlinear-provider", "openai-oauth")
    code, status = cli(capsys, str(images), "status")
    assert code == 0
    assert status["result"]["image_model"] == {
        "provider": "openai-oauth", "connected": False, "trace_borders": False}
    assert "trace_borders" not in status["result"]["verbs"]
    code, launch = cli(capsys, str(images), "launch", "--role", "register-scripting")
    from langslice.tools.descriptions.door_text import IMAGE_MODEL_OFF

    assert IMAGE_MODEL_OFF in launch["result"]["prompt"]
    from langslice.doors import credentials as setup

    monkeypatch.setattr(setup, "image_model_connected", lambda provider: True)
    code, status = cli(capsys, str(images), "status")
    assert status["result"]["image_model"]["connected"] is True
    assert "trace_borders" in status["result"]["verbs"]


def test_status_marks_what_the_user_froze_or_marked_damaged(capsys, images):
    init(capsys, images, "--frozen", json.dumps({ID0: ["orientation", "affine"]}),
         "--damaged", json.dumps({ID1: "torn"}))
    code, envelope = cli(capsys, str(images), "status")
    assert code == 0
    rows = {row["id"]: row for row in envelope["result"]["rows"]}
    assert rows[ID0]["frozen"] == ["orientation", "affine"]
    assert rows[ID0]["damage_by_user"] is False
    assert rows[ID1]["damage_by_user"] is True and rows[ID1]["frozen"] == []
    assert rows[ID2]["frozen"] == [] and rows[ID2]["damage_by_user"] is False
    assert rows[ID1]["damage_note"] == "torn" and rows[ID1]["damaged"] is False
    assert rows[ID1]["damaged_regions"] == []


def test_every_call_is_logged_and_traced(capsys, images, monkeypatch, tmp_path):
    traces = tmp_path / "traces"
    monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(traces))
    job = init(capsys, images)
    code, envelope = cli(capsys, str(images), "status", "--verbose")
    assert code == 0
    code, refused = cli(capsys, str(images), "export_maps", "--sections", "nope.png")
    assert code == 2
    lines = [json.loads(line) for line in (job / "logs" / "calls.jsonl").read_text(
        encoding="utf-8")
             .splitlines()]
    assert [line["verb"] for line in lines] == ["init", "status", "export_maps"]
    seen = lines[1]
    assert seen["arguments"] == ["--verbose"] and seen["ok"] is True
    assert seen["exit"] == 0 and seen["artifacts"] == []
    assert lines[2]["ok"] is False and lines[2]["error"] == "UNKNOWN_SLICE_IDS"
    assert [line["reply"] for line in lines] == [
        "logs/calls/000001_init.json", "logs/calls/000002_status.json",
        "logs/calls/000003_export_maps.json"]
    assert json.loads((job / lines[1]["reply"]).read_text(
        encoding="utf-8")) == envelope
    assert json.loads((job / lines[2]["reply"]).read_text(
        encoding="utf-8")) == refused
    assert lines[2]["message"] == refused["error"]["message"]
    files = list(traces.glob("cli_stack_*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text(
        encoding="utf-8").splitlines()]
    assert [record["kind"] for record in records] == ["tool_result"] * 3
    traced = records[1]
    assert traced["name"] == "status"
    assert json.loads(traced["content"][0]["text"])["ok"] is True


def _tool_events(job: Path) -> list[dict[str, Any]]:
    lines = [json.loads(line) for line in (job / "logs" / "events.jsonl").read_text(
        encoding="utf-8")
             .splitlines()]
    return [line for line in lines if line["kind"] in ("tool_start", "tool_end")]


def test_every_tool_call_is_an_event_in_the_job_folder(capsys, images):
    """Each call on the CLI door and the library door: a `tool_start` and a
    `tool_end` line in logs/events.jsonl, the pictures as job-relative paths,
    never the reply; a dry run logs nothing."""
    import langslice

    job = init(capsys, images)
    code, _ = cli(capsys, str(images), "status")
    assert code == 0
    with langslice.open_job(images) as handle:
        handle.look(mode="section", sections=[ID0])
        handle.position_sections(sections=[{"id": ID1, "position_mm": 0.15}])
        assert handle.redo()["status"] == "error"
    code, _ = cli(capsys, str(images), "export_maps", "--sections", ID1, "--dry-run")
    assert code == 0

    events = _tool_events(job)
    assert [(e["kind"], e["name"], e["door"]) for e in events] == [
        ("tool_start", "status", "cli"), ("tool_end", "status", "cli"),
        ("tool_start", "look", "library"), ("tool_end", "look", "library"),
        ("tool_start", "position_sections", "library"),
        ("tool_end", "position_sections", "library"),
        ("tool_start", "redo", "library"), ("tool_end", "redo", "library"),
    ]
    for start, end in zip(events[::2], events[1::2], strict=True):
        assert start["execution_id"] == end["execution_id"] and start["at"]
        assert "response" not in end and "response" not in start
    look_start, look_end = events[2], events[3]
    assert look_start["target_ids"] == [ID0]
    assert look_start["args"]["sections"] == [ID0] and look_start["args"]["mode"] == "section"
    assert look_end["failed"] is False and look_end["status"] == "ok"
    assert look_end["step"] == 0
    assert look_end["views"] and all(
        not Path(view).is_absolute() and (job / view).is_file() for view in look_end["views"])
    write_end = events[5]
    assert write_end["target_ids"] == [ID1] and write_end["step"] == 1
    assert write_end["failed"] is False
    redo_end = events[7]
    assert redo_end["failed"] is True and redo_end["status"] == "error"


def test_the_gui_watcher_follows_real_calls_on_the_cli_and_library_doors(
        capsys, images, tmp_path, monkeypatch):
    """End to end, nothing synthesized: `init` lists the job in the recent jobs
    file, the GUI's job list finds it there, and its watcher turns the lines
    real CLI and library calls wrote into activity rows whose pictures are the
    pictures the calls saved, with the write labelling its undo step."""
    import langslice
    from langslice.hosts.gui.watch import JobList

    recent = tmp_path / "recent_jobs.jsonl"
    monkeypatch.setenv("LANGSLICE_RECENT_JOBS", str(recent))
    job = init(capsys, images)
    jobs = JobList([], recent=recent)
    assert [Path(entry["folder"]) for entry in jobs.listing()] == [job]
    watcher = jobs.get(0)
    assert watcher is not None

    code, _ = cli(capsys, str(images), "status")
    assert code == 0
    with langslice.open_job(images) as handle:
        looked = handle.look(mode="section", sections=[ID0])
        handle.position_sections(sections=[{"id": ID1, "position_mm": 0.15}])
        handle.look(mode="section", sections=[ID2])

    watcher.poll(force=True)
    snapshot = watcher.snapshot(since=0, generation=watcher.generation)
    tools = [item for item in snapshot["items"] if item["kind"] == "tool"]
    assert [(t["name"], t["phase"], t["door"]) for t in tools] == [
        ("status", "start", "cli"), ("status", "end", "cli"),
        ("look", "start", "library"), ("look", "end", "library"),
        ("position_sections", "start", "library"), ("position_sections", "end", "library"),
        ("look", "start", "library"), ("look", "end", "library")]
    look_end = tools[3]
    assert look_end["sections"] == [ID0] and look_end["failed"] is False
    assert look_end["args"]["sections"] == [ID0]
    pictures = {item["path"]: item for item in snapshot["items"] if item["kind"] == "picture"}
    assert look_end["views"] and all(view in pictures for view in look_end["views"])
    shown = watcher.picture(pictures[look_end["views"][0]]["number"])
    assert shown is not None
    view_artifact = next(item for item in looked["artifacts"] if item["kind"] == "view")
    assert shown.resolve() == Path(view_artifact["path"]).resolve()
    assert tools[5]["step"] == 1 and tools[7]["views"]
    assert watcher.undo()["undo"] == [f"position_sections ({ID1})"]
    assert snapshot["status"]["word"] == "Running"



def test_exit_codes_and_fix_hints(capsys, images):
    init(capsys, images)
    code, envelope = cli(capsys, str(images), "export_maps", "--sections", "[]",
                         "--bogus", "1")
    assert code == 2 and envelope["ok"] is False
    assert envelope["error"]["code"] == "UNKNOWN_ARGUMENTS"
    job = images / "langslice"
    fix = envelope["error"]["fix"]
    assert fix.startswith("export_maps takes sections (")
    assert f"Call it as: `langslice-job {job} export_maps --sections '[]'" in fix
    code, envelope = cli(capsys, str(images), "export_maps", "--sections", "nope.png")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_SLICE_IDS"
    code, envelope = cli(capsys, str(images), "export_maps", "--args", "{not json")
    assert code == 2 and envelope["error"]["code"] == "BAD_JSON"
    code, envelope = cli(capsys, str(images), "fit_affine")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_VERB"


def test_an_internal_error_exits_4(capsys, images, monkeypatch):
    init(capsys, images)
    from langslice.doors.agent_cli import calls

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(calls, "shape", broken)
    code, envelope = cli(capsys, str(images), "status")
    assert code == 4 and envelope["error"]["code"] == "INTERNAL"
    assert "boom" in envelope["error"]["message"]


def test_a_first_sentence_ends_at_a_sentence_not_an_abbreviation():
    from langslice.doors.agent_cli.calls import _first_sentence

    assert _first_sentence('A glob (e.g. "*.json"): every file. Empty lists it.') == (
        'A glob (e.g. "*.json"): every file.')
    assert _first_sentence("Steps, e.g. three. More.") == "Steps, e.g. three."
    assert _first_sentence("No full stop") == "No full stop"


def test_folder_ops_lists_the_job_s_own_verbs(capsys, images):
    job = init(capsys, images, "--tasks", "position", "--disabled", "export_maps")
    code, envelope = cli(capsys, str(images), "ops")
    assert code == 0, envelope
    names = [row["name"] for row in envelope["result"]["verbs"]]
    assert envelope["result"]["job"] == str(job)
    assert "status" in names and "export_maps" not in names
    assert envelope["next"][0] == f"langslice-job {job} schema VERB [VERB ...]"
    code, envelope = cli(capsys, str(images), "ops", "--bogus")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGS"


def test_a_failed_fit_says_the_first_line_and_keeps_the_rest_in_the_log(tmp_path):
    from langslice.core.fits.engine_log import failure_message

    log = tmp_path / "logs" / "engine.log"
    text = ("AcceleratorError: CUDA error: out of memory\nSearch for "
            "`cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/...\nmore lines")
    words = failure_message(text, log, "the fit of s0.png failed")
    assert words.startswith("AcceleratorError: CUDA error: out of memory. The GPU ran out of "
                            "memory")
    assert "cudaErrorMemoryAllocation" not in words
    assert words.endswith("logs/engine.log holds the whole error.")
    held = log.read_text(encoding="utf-8")
    assert "the fit of s0.png failed" in held and "more lines" in held
    assert failure_message("ValueError: bad", None) == "ValueError: bad."


def test_export_maps_writes_the_derived_files_as_artifacts(capsys, images):
    job = init(capsys, images)
    place(images)
    code, envelope = cli(capsys, str(images), "export-maps", "--sections", ID0,
                         "--dry-run")
    assert code == 0 and envelope["result"]["files_written"] is False
    assert envelope["artifacts"] == [] and not (job / "sections/s0/coords.tif").exists()
    assert envelope["next"] == [f"langslice-job {job} export_maps --args "
                                f"'{json.dumps({'sections': [ID0]})}'"]
    code, envelope = cli(capsys, str(images), "export_maps", "--sections", ID0,
                         "--dry-run", "--background")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGS"
    assert not (job / "logs" / "runs").exists()
    code, envelope = cli(capsys, str(images), "export_maps", "--sections", ID0)
    assert code == 0, envelope
    assert envelope["result"]["written"] == [ID0] and envelope["result"]["n_files"] >= 5
    kinds = {artifact["kind"] for artifact in envelope["artifacts"]}
    assert {"coords", "labels", "labels_fiji", "labels_csv", "maps", "quicknii",
            "visualign", "registration"} <= kinds
    assert all(Path(artifact["path"]).is_file() for artifact in envelope["artifacts"])
    code, envelope = cli(capsys, str(images), "status")
    assert "export_maps" in envelope["result"]["verbs"]



def test_a_background_run_answers_at_once_then_status_and_wait(capsys, images, monkeypatch):
    from langslice.doors.agent_cli import background

    init(capsys, images)
    place(images)
    assert background.CHILD_COMMAND[-1] == "tests.cli_child"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO / "src"), str(REPO))))
    monkeypatch.chdir(REPO)
    code, envelope = cli(capsys, str(images), "export_maps", "--sections", ID0,
                         "--background")
    assert code == 0 and envelope["result"]["state"] == "running"
    run = envelope["result"]["run"]
    assert envelope["next"] == [f"langslice-job {images / 'langslice'} wait {run}"]
    code, waited = cli(capsys, str(images), "wait", run, "--timeout", "120")
    assert code == 0, waited
    assert waited["result"]["run"] == run
    assert waited["result"]["written"] == [ID0]
    assert "coords" in {artifact["kind"] for artifact in waited["artifacts"]}
    assert Path(waited["artifacts"][0]["path"]).is_file()
    code, one = cli(capsys, str(images), "runs", run)
    assert code == 0 and one["result"]["state"] == "finished"
    assert one["result"]["exit"] == 0
    code, listed = cli(capsys, str(images), "runs")
    assert listed["result"]["runs"][0]["id"] == run
    code, unknown = cli(capsys, str(images), "runs", "no-such-run")
    assert code == 2 and unknown["error"]["code"] == "UNKNOWN_RUN"
    code, status = cli(capsys, str(images), "status", run)
    assert code == 2 and status["error"]["code"] == "BAD_ARGS"


def test_wait_answers_still_running_within_its_default_wait(capsys, images, monkeypatch):
    """A harness ends a command it finds too long, so `wait` answers by itself
    (``background.WAIT_S``, 30 s) with the run still running and how to wait
    again: an answer (ok, exit 0), not an error."""
    import time

    from langslice.doors.agent_cli import background
    from langslice.job.folder.layout import JobLayout

    job = init(capsys, images)
    layout = JobLayout(job)
    run_id = "20261021-000000-abcdef"
    background._write(layout, {"id": run_id, "verb": "fireants_syn", "arguments": {},
                               "state": "running", "started_at": "2026-10-21T00:00:00+00:00",
                               "started": time.time(), "pid": os.getpid()})
    monkeypatch.setattr(background, "WAIT_S", 0.5)
    started = time.monotonic()
    code, envelope = cli(capsys, str(images), "wait", run_id)
    assert time.monotonic() - started < 10
    assert code == 0 and envelope["ok"] is True and "error" not in envelope
    assert envelope["result"]["state"] == "running" and envelope["result"]["id"] == run_id
    assert envelope["result"]["waited_s"] == 0.5
    assert "Still running after 0.5 s" in envelope["result"]["message"]
    assert envelope["next"] == [f"langslice-job {job} wait {run_id}"]
    started = time.monotonic()
    code, envelope = cli(capsys, str(images), "wait", run_id, "--timeout", "1.5")
    assert code == 0 and time.monotonic() - started >= 1.5
    assert envelope["result"]["waited_s"] == 1.5
    background._write(layout, {"id": run_id, "verb": "fireants_syn", "arguments": {},
                               "state": "running", "started_at": "2026-10-21T00:00:00+00:00",
                               "started": time.time(), "pid": 2 ** 22 + 12345})
    code, lost = cli(capsys, str(images), "wait", run_id)
    assert code == 4 and lost["ok"] is False and lost["error"]["code"] == "RUN_LOST"


def test_detaching_follows_the_setting_and_claude_code(monkeypatch):
    """A long verb's foreground call runs detached unless LANGSLICE_DETACH says
    no, or, unset, under Claude Code, whose sandbox ends what a command leaves."""
    from langslice.doors.agent_cli.background import DETACH_ENV, detaching

    monkeypatch.delenv(DETACH_ENV, raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    assert detaching()
    monkeypatch.setenv("CLAUDECODE", "1")
    assert not detaching()
    monkeypatch.setenv(DETACH_ENV, "1")
    assert detaching()
    monkeypatch.setenv(DETACH_ENV, "0")
    monkeypatch.delenv("CLAUDECODE")
    assert not detaching()


def test_a_long_verb_runs_detached_and_answers_within_its_wait(capsys, images, monkeypatch):
    """Where processes may outlive their command, a long verb called without
    --background runs as a background run: the call answers with its result
    when it lands in time, else with its run id and state running (the run
    goes on, and `wait` collects it)."""
    from langslice.doors.agent_cli import background
    from langslice.job.folder.layout import JobLayout

    init(capsys, images)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO / "src"), str(REPO))))
    monkeypatch.chdir(REPO)
    place(images, ID0, ID1)
    monkeypatch.setenv(background.DETACH_ENV, "1")
    monkeypatch.setattr(background, "WAIT_S", 120.0)
    code, exported = cli(capsys, str(images), "export_maps", "--sections", ID0)
    assert code == 0, exported
    run = exported["result"]["run"]
    assert exported["result"]["written"] == [ID0]
    assert all(Path(artifact["path"]).is_file() for artifact in exported["artifacts"])
    assert background.read(JobLayout(images / "langslice"), run)["state"] == "finished"
    monkeypatch.setattr(background, "WAIT_S", 0.01)
    code, running = cli(capsys, str(images), "export_maps", "--sections", ID1)
    assert code == 0 and running["ok"] is True, running
    assert running["result"]["state"] == "running"
    later = running["result"]["id"]
    assert running["next"] == [f"langslice-job {images / 'langslice'} wait {later}"]
    code, landed = cli(capsys, str(images), "wait", later, "--timeout", "120")
    assert code == 0 and landed["result"]["run"] == later
    assert landed["result"]["written"] == [ID1]



def test_a_script_and_a_running_toolbox_interleave_on_one_folder(capsys, images):
    import langslice
    from langslice.doors.jobs import context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import Job
    from langslice.job.settings.spec import (
        JobSpec,
        NonlinearSpec,
        PositionSpec,
    )

    job_folder = init(capsys, images)
    code, envelope = cli(capsys, str(images), "status")
    start = {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]}
    spec = JobSpec(image_folder=str(images), preprocess="none", resume=True,
                   tasks=["position", "transform", "nonlinear"],
                   inputs={"pixel_size_um": PIXEL_SIZE_UM}, position=PositionSpec(),
                   nonlinear=NonlinearSpec(provider="none"))
    ctx = context(spec, job_folder)
    agent = Job.open(spec, ctx, folder=job_folder, results_path=ctx.results_path)
    tools = {tool.__name__: tool for tool in build_tools(agent.state, ctx, spec,
                                                         job=agent).tools}
    assert tools["position_sections"](sections=[{"id": ID0, "position_mm": 0.1}],
                                      view=False)["status"] == "ok"

    script = langslice.open_job(images)
    assert script.position_sections(
        sections=[{"id": ID1, "position_mm": 0.15}])["status"] == "ok"
    code, envelope = cli(capsys, str(images), "status")
    positions = {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]}
    assert positions == {ID0: 0.1, ID1: 0.15, ID2: start[ID2]}

    rows = tools["status"]()["rows"]
    assert {row["id"]: row.get("position_mm") for row in rows} == positions
    assert tools["undo"]()["status"] == "ok"
    assert agent.state.resolve(ID1).position_mm == start[ID1]
    assert agent.state.resolve(ID0).position_mm == 0.1

    assert script.undo()["status"] == "ok"
    script.close()
    code, envelope = cli(capsys, str(images), "status")
    assert {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]} == start
    assert tools["status"]()["rows"][0].get("position_mm") == start[ID0]
    agent.close()



def test_open_job_gives_the_verbs_as_methods(capsys, images):
    import langslice

    init(capsys, images)
    with langslice.open_job(images) as job:
        assert "position_sections" in job.verbs and "position_sections" in dir(job)
        reply = job.position_sections(sections=[{"id": ID2, "position_mm": 0.2}])
        assert reply["status"] == "ok" and "images" not in reply
        assert json.loads(json.dumps(reply)) == reply
        views = [item for item in reply["artifacts"] if item["kind"] == "view"]
        assert views and all(Path(item["path"]).is_file() for item in reply["artifacts"])
        assert [item["index"] for item in views] == list(range(len(reply.images)))
        assert job.state.resolve(ID2).position_mm == 0.2
        rows = job.status()["rows"]
        assert len({tuple(row) for row in rows}) == 1
        assert rows[-1]["delta_to_next_mm"] is None
        with pytest.raises(AttributeError, match="This job's settings have no 'trace_borders'"):
            getattr(job, "trace_borders")  # noqa: B009
        overlay = job.look(mode="overlay", sections=[ID2])
        saved = next(item["path"] for item in overlay["artifacts"]
                     if item["kind"] == "view_json")
    code, envelope = cli(capsys, str(images), "status")
    assert envelope["result"]["rows"][2]["position_mm"] == 0.2
    xyz = langslice.coordinate_map(saved)
    assert xyz.ndim == 3 and xyz.shape[2] == 3 and bool((xyz == xyz).any())



def _dead_run(tmp_path: Path) -> tuple[Any, str]:
    from langslice.doors.agent_cli import background
    from langslice.job.folder.layout import JobLayout

    layout = JobLayout(tmp_path / "job")
    background._write(layout, {"id": "run-1", "verb": "fireants_syn", "state": "running",
                               "started": 0.0, "pid": 424242})
    return layout, "run-1"


def test_a_dead_run_reads_lost_on_windows(tmp_path, monkeypatch):
    """On Windows the run's process is really probed, and ``wait`` without a
    timeout returns once the process is gone."""
    import sys
    import threading
    import types

    from langslice.doors.agent_cli import background

    monkeypatch.setattr(background, "WINDOWS", True)
    fake = types.SimpleNamespace(pid_exists=lambda pid: pid != 424242)
    monkeypatch.setitem(sys.modules, "psutil", fake)
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


def test_the_library_caps_no_pictures_or_sections_but_fireants_syn(capsys, images):
    """``Verb.limits`` (what the card lists) holds the compute cap only; the
    pictures and the transform sections of one call are not capped."""
    import langslice
    from langslice.tools.catalog import VERBS

    init(capsys, images)
    limits = {name: dict(verb.limits) for name, verb in VERBS.items() if verb.limits}
    assert limits == {"fireants_syn": {"sections": limits["fireants_syn"]["sections"]}}
    with langslice.open_job(images) as job:
        placed = job.position_sections(
            sections=[{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
                      {"id": ID2, "position_mm": 0.2}], view=False)
        assert placed["status"] == "ok", placed
        most = limits["fireants_syn"]["sections"]
        refused = job.fireants_syn(sections=[ID0] * (most + 1))
        assert refused["error"] == "TOO_MANY_SECTIONS" and refused["max_sections"] == most
        assert most == 20 and refused["message"] == (
            "fireants_syn takes at most 20 sections per call (every fitted section is held in "
            "memory (about 170 MB) until the call writes them all as one undo step); 21 were "
            "named. Nothing was done: split them over calls.")
        sections = [ID0, ID1, ID2] * 2
        written = job.interactive_transform(
            sections=[{"id": name, "rotation_deg": 0} for name in sections[:3]] * 2, view=False)
        assert written.get("error") != "TOO_MANY_SECTIONS"
        positions = [0.02 + 0.04 * i for i in range(7)]
        shown = job.look(mode="atlas", positions_mm=positions)
        assert len(shown.images) == 7 and "not_shown" not in shown
        shown = job.grep_atlas_view(regions=["root"], positions_mm=positions)
        assert len(shown.images) == 7 and "not_shown" not in shown


def test_a_transform_cap_of_the_job_holds_without_a_picture_cap():

    from langslice.core.refusal import Refused
    from langslice.tools.refusal import check_section_count

    check_section_count(SimpleNamespace(picture_cap=None, transform_cap=None),
                        "elastix_affine", 50)
    check_section_count(SimpleNamespace(picture_cap=4, transform_cap=None),
                        "interactive_transform", 4)
    for job in (SimpleNamespace(picture_cap=None, transform_cap=2),
                SimpleNamespace(picture_cap=4, transform_cap=2)):
        with pytest.raises(Refused) as caught:
            check_section_count(job, "elastix_affine", 3)
        assert caught.value.code == "TOO_MANY_SECTIONS"
    with pytest.raises(Refused):
        check_section_count(SimpleNamespace(picture_cap=4, transform_cap=None),
                            "interactive_transform", 5)


def test_the_door_sets_the_job_picture_cap(tmp_path):
    from tests.linear_tool_helpers import box

    _state, _spec, adk = box(tmp_path, placed=True)
    assert adk.job.picture_cap == 4


_EXIT_SCRIPT = """
import sys
from tests.cli_child import install
from tests.synthetic_stack import ID0, ID1, ID2, apply_patches, atlas_loader
apply_patches()
install(atlas_loader())
import langslice
job = langslice.open_job(sys.argv[1])
job.position_sections(sections=[{"id": name, "position_mm": 0.05 + index * 0.05}
                                for index, name in enumerate([ID0, ID1, ID2])], view=False)
raw = {tool.__name__: tool for tool in job._opened.tools().tools}
# The tool itself, which only queues its pictures: the writer is still busy
# when the script ends, and no one calls close().
raw["interactive_transform"](sections=[{"id": name, "rotation_deg": 1.0 + index}
                                       for index, name in enumerate([ID0, ID1, ID2])])
"""


def test_pictures_queued_by_a_script_are_written_when_it_exits(capsys, images):
    """At exit the picture writer still runs (its TIFF encoder starts
    threads), so a script that never closes its job keeps its pictures."""
    import subprocess
    import sys

    job = init(capsys, images)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, "-c", _EXIT_SCRIPT, str(images)], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert "cannot schedule new futures" not in done.stderr and "Could not save" not in done.stderr
    lines = [json.loads(line) for line in (job / "views.jsonl").read_text(
        encoding="utf-8").splitlines()]
    placed = [line for line in lines if line["tool"] == "interactive_transform"]
    assert len(placed) == 3 and all(line["layers"] for line in placed)
    for line in placed:
        assert (job / line["path"] / "labels.tif").is_file()



def test_disable_takes_tools_and_options_off_a_cli_job(capsys, images):
    """``--disable``: the verb is not offered, an option is not in the schema,
    a call of it is refused, and nothing the agent reads names either."""
    code, envelope = cli(capsys, str(images), "init", "--tasks", "position,transform",
                         "--nonlinear-provider", "none", "--preprocess", "none",
                         "--pixel-size-um", str(PIXEL_SIZE_UM), "--disabled", "zoom",
                         "--disabled", "look:positioning,interactive_transform:flip")
    assert code == 0, envelope
    folder = Path(envelope["result"]["job_folder"])
    assert "zoom" not in envelope["result"]["verbs"] and "look" in envelope["result"]["verbs"]
    assert json.loads((folder / "job.json").read_text(
        encoding="utf-8"))["spec"]["disabled"] == [
        "zoom", "look:positioning", "interactive_transform:flip"]
    code, launched = cli(capsys, str(folder), "launch", "--role", "register-scripting")
    assert code == 0, launched
    statement = launched["result"]["prompt"]
    assert "`zoom`" not in statement and "mode positioning" not in statement
    card = (folder / "AGENTS.md").read_text(encoding="utf-8")
    assert "- `zoom`" not in card and "- `look`" in card
    assert (folder / "CLAUDE.md").read_text(encoding="utf-8") == card

    import langslice

    with langslice.open_job(folder) as job:
        assert "zoom" not in job.verbs
        assert job.look(mode="positioning")["error"] == "OPTION_OFF"
    code, schema = cli(capsys, "schema", "look", "--job", str(folder))
    assert schema["result"]["arguments"]["properties"]["mode"]["enum"] == [
        "section", "atlas", "overlay"]
    code, off = cli(capsys, "schema", "zoom", "--job", str(folder))
    assert code == 3 and off["error"]["code"] == "VERB_OFF"
    code, everything = cli(capsys, "schema", "--job", str(folder))
    assert "zoom" not in everything["result"]["verbs"]
    code, sections = cli(capsys, "schema", "interactive_transform", "--job", str(folder))
    defs = sections["result"]["arguments"]["$defs"]
    assert all("flip" not in entry["properties"] for entry in defs.values())


def test_a_bad_disable_ends_init_with_the_valid_names(capsys, images):
    code, envelope = cli(capsys, str(images), "init", "--disabled", "zoomm")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGS"
    assert "unknown name 'zoomm'" in envelope["error"]["message"]
    assert "look:positioning" in envelope["error"]["message"]
    code, envelope = cli(capsys, str(images), "init", "--tasks", "transform",
                         "--disabled", "elastix_affine,interactive_transform")
    assert code == 2 and "every tool that writes its answer" in envelope["error"]["message"]



def test_the_command_line_runs_only_the_setup_and_checking_verbs(capsys, images):
    """`langslice-job FOLDER VERB` runs `status` and `export_maps` and refuses
    a registration tool, saying where it is; `ops` lists what it runs, and
    `schema` still describes every verb (the library's methods take the same arguments)."""
    from langslice.doors.agent_cli.catalog import CLI_VERBS

    job = init(capsys, images)
    assert CLI_VERBS == {"status", "export_maps", "review_comments"}

    code, status = cli(capsys, str(job), "status")
    assert code == 0 and len(status["result"]["rows"]) == 3
    code, exported = cli(capsys, str(job), "export_maps")
    assert code == 0 and exported["ok"] is True, exported

    for verb, arguments in [("position_sections", {"sections": [
            {"id": ID0, "position_mm": 0.1}]}), ("look", {"mode": "section"}),
            ("undo", {}), ("submit", {"summary": "done"})]:
        code, refused = cli(capsys, str(job), verb, "--args", json.dumps(arguments))
        assert code == 3 and refused["error"]["code"] == "VERB_OFF", refused
        assert "registration tool, not a command" in refused["error"]["message"]
        assert f"open_job(FOLDER).{verb}(" in refused["error"]["fix"]
    code, after = cli(capsys, str(job), "status")
    assert all(row["position_source"] == "initial" for row in after["result"]["rows"])

    code, listed = cli(capsys, str(job), "ops")
    assert code == 0 and {row["name"] for row in listed["result"]["verbs"]} == set(CLI_VERBS)
    assert "launch --role register|register-scripting" in listed["result"]["job_commands"]
    code, described = cli(capsys, str(job), "schema", "position_sections")
    assert code == 0 and "sections" in described["result"]["arguments"]["properties"]
    assert "open_job" in described["next"][0]
