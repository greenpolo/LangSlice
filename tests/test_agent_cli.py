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

from langslice.doors.cli.jobcli import main
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


def init(capsys: pytest.CaptureFixture[str], images: Path, *flags: str) -> Path:
    code, envelope = cli(capsys, str(images), "init", "--tasks",
                         "position,transform,nonlinear", "--image-provider", "none",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         *flags)
    assert code == 0, envelope
    return Path(envelope["result"]["job_folder"])


# --- ops and schema -----------------------------------------------------------------


def test_schema_of_one_verb_and_of_every_verb(capsys):
    from langslice.doors.cli.catalog import SCHEMA_VERSION
    from langslice.doors.declarations import FULL, Variant, declaration

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
    # What a model reads of the verb: the whole description, the summary, long
    # or not. The picture options are gone with the view modes.
    assert result["description"] == declaration(
        "interactive_transform",
        Variant(auto=True, door="cli", uncapped=True, prompt=FULL.prompt)).doc
    assert "\n" in result["description"] and result["summary"] in result["description"]
    assert "picture_options" not in result
    assert result["long"] is False and result["job"] is None and "--job" in result["hint"]
    # The CLI's caller sizes each picture: look takes resolution.
    code, look = cli(capsys, "schema", "look")
    assert code == 0 and "resolution" in look["result"]["arguments"]["properties"]
    code, fit = cli(capsys, "schema", "elastix_affine")
    # A long verb's call is safe in the foreground: no --background suggested.
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
    # Without a job, a name that may be a job's plugin verb says how to name the job.
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
    code, refused = cli(capsys, "schema", "look", "spomialign")
    assert code == 2 and refused["error"]["code"] == "UNKNOWN_VERB"
    assert "'spomialign'" in refused["error"]["message"]


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

    from langslice.core.atlas.core import load_atlas

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
    # The positions are the host's; the agent sets only the cutting angles.
    code, envelope = cli(capsys, str(images), "init", "--tasks", "transform", "--angles",
                         "--image-provider", "none", "--preprocess", "none",
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
    # Run in a job folder, schema declares that job's verbs.
    monkeypatch.chdir(images / "langslice")
    code, here = cli(capsys, "schema", "position_sections")
    assert here["result"]["job"] == str(images / "langslice")
    assert "sections" not in here["result"]["arguments"]["properties"]
    # Only the job's verbs: no Nonlinear verb for a transform job.
    code, listed = cli(capsys, "schema", "--job", str(images))
    assert "fireants_syn" not in listed["result"]["verbs"]
    assert "interactive_transform" in listed["result"]["verbs"]
    code, off = cli(capsys, "schema", "fireants_syn", "--job", str(images))
    assert code == 3 and off["error"]["code"] == "VERB_OFF"


def test_schema_of_an_unreadable_job_is_one_envelope(capsys, images):
    job = init(capsys, images)
    record = json.loads((job / "job.json").read_text())
    record["spec"]["plane"] = "diagonal"  # a spec the job layer refuses
    (job / "job.json").write_text(json.dumps(record))
    code, envelope = cli(capsys, "schema", "fireants_syn", "--job", str(images))
    assert code == 3 and envelope["error"]["code"] == "JOB_UNREADABLE"


# --- init, the card, status ---------------------------------------------------------


def test_init_creates_the_job_and_its_reference_card(capsys, images):
    job = init(capsys, images)
    assert job == images / "langslice"
    agents = (job / "AGENTS.md").read_text()
    assert agents == (job / "CLAUDE.md").read_text()
    from langslice.ops.deformable import MAX_FIREANTS_SYN_SECTIONS
    from langslice.ops.registry import VERBS

    for needle in ("state.json", "coordinate_map", "langslice-job", "langslice.open_job",
                   "i * resolution", " brief`", "BRIEF.md", "STALE_INPUT",
                   "- `elastix_affine` (write, Linear, long):",
                   "- `look` (read, Common):",
                   "- `interactive_transform` (write, Linear):",
                   f"- `fireants_syn` (write, Nonlinear, long; at most {MAX_FIREANTS_SYN_SECTIONS} "
                   "sections per call):",
                   'job.position_sections(sections=[{"id": "<file>", "position_mm": 5.2}])',
                   "schema init", "reply.images", "`scripts/`", "never in /tmp",
                   "`env install PACKAGE ...`", "`job.context`"):
        assert needle in agents, needle
    assert len(agents.splitlines()) < 90  # one screen, its verb list included
    code, envelope = cli(capsys, str(images), "status")
    assert code == 0
    # The card lists exactly the verbs the job has: none it would refuse.
    listed = [name for name in VERBS if f"- `{name}` (" in agents]
    assert listed == [name for name in VERBS if name in envelope["result"]["verbs"]]
    assert [row["id"] for row in envelope["result"]["rows"]] == [ID0, ID1, ID2]
    assert {"position_sections", "elastix_affine", "fireants_syn", "export_maps"} <= set(
        envelope["result"]["verbs"])
    # Opening the job folder itself works too; a folder without a job is refused.
    code, _envelope = cli(capsys, str(job), "status")
    assert code == 0
    code, missing = cli(capsys, str(images.parent), "status")
    assert code == 2 and missing["error"]["code"] == "NO_JOB"
    # A stale card is rewritten on the next open.
    (job / "AGENTS.md").write_text("old")
    cli(capsys, str(images), "status")
    assert (job / "AGENTS.md").read_text() == agents


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
    code, envelope = cli(capsys, str(images), "brief")
    assert code == 0, envelope
    result = envelope["result"]
    statement = result["statement"]
    code, status = cli(capsys, str(images), "status")
    verbs = status["result"]["verbs"]  # this door's verbs: the scripting ones too
    assert "export_maps" in verbs
    native, pictures, opened = _native(images, DEFAULT_IMAGE_LIMIT)
    try:
        # The same text as the native agent's, but for the tools this door
        # offers, then where this door's opening pictures are.
        tools = next(line for line in native.splitlines() if line.startswith("Tools ("))
        expected = native.replace(tools, tools.split(": ", 1)[0] + ": "
                                  + ", ".join(f"`{name}`" for name in verbs) + ".")
        assert expected != native and statement.startswith(expected + "\n\n")
        tail = statement[len(expected):]
        assert "Opening pictures: `brief` saved 2 picture files" in tail
        assert "User notes:\nSection 2 is torn." in tail
        from langslice.doors.statement import status_and_notes

        assert tail.endswith(status_and_notes(opened.job.state))
    finally:
        opened.close()
    # The opening pictures: the native strips, byte for byte, in reading order.
    opening = [item for item in envelope["artifacts"] if item["kind"] == "opening"]
    assert [item["index"] for item in opening] == list(range(len(pictures)))
    assert [Path(item["path"]).read_bytes() for item in opening] == pictures
    assert opening[0]["label"].startswith("Strip 1 of 1: s0.png")
    assert [entry.get("path") for entry in result["opening"] if "picture" in entry] == [
        item["path"] for item in opening]
    # BRIEF.md holds it all, and the card names it first.
    brief = (job / "BRIEF.md").read_text(encoding="utf-8")
    assert statement.replace("\r\n", "\n") in brief
    assert all(item["path"] in brief for item in opening)
    assert envelope["artifacts"][-2] == {"path": str(job / "BRIEF.md"), "kind": "brief"}
    assert "BRIEF.md" in (job / "AGENTS.md").read_text()
    # TOOLS.md beside it holds every verb's description and arguments, as declared here.
    assert envelope["artifacts"][-1] == {"path": str(job / "TOOLS.md"), "kind": "tools"}
    tools = (job / "TOOLS.md").read_text(encoding="utf-8")
    assert "## look" in tools and "## export_maps" in tools and "- `mode`: one of" in tools
    assert "TOOLS.md" in (job / "AGENTS.md").read_text()
    assert result["viewer"] == "codex" and result["resolution"]["max"] == 2048


def test_init_answers_with_the_statement_and_keeps_notes_and_viewer(capsys, images):
    code, envelope = cli(capsys, str(images), "init", "--tasks", "position",
                         "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         "--notes", "Mind the bulbs.")
    assert code == 0, envelope
    statement = envelope["result"]["statement"]
    assert "User notes:\nMind the bulbs." in statement
    assert envelope["result"]["viewer"] == "claude"  # the default viewer
    assert envelope["next"] == [f"langslice-job {images / 'langslice'} brief"]
    record = json.loads((images / "langslice" / "job.json").read_text())
    assert record["notes"] == "Mind the bulbs." and "viewer" not in record
    # Continuing the job keeps the notes; --viewer is stored.
    code, again = cli(capsys, str(images), "init", "--tasks", "position",
                      "--preprocess", "none", "--pixel-size-um", str(PIXEL_SIZE_UM),
                      "--viewer", "openai")
    assert code == 0 and again["result"]["viewer"] == "openai"
    record = json.loads((images / "langslice" / "job.json").read_text())
    assert record["notes"] == "Mind the bulbs." and record["viewer"] == "openai"


def test_init_on_an_existing_job_refuses_other_settings(capsys, images):
    """DOORS-2: a second init with other job flags does not replace the saved settings."""
    job = init(capsys, images)
    before = json.loads((job / "job.json").read_text())["spec"]
    code, envelope = cli(capsys, str(images), "init", "--pixel-size-um", str(PIXEL_SIZE_UM),
                         "--notes", "Mind the bulbs.")
    assert code != 0 and envelope["error"]["code"] == "SETTINGS_CHANGED", envelope
    assert "tasks" in envelope["error"]["message"]
    assert json.loads((job / "job.json").read_text())["spec"] == before


def test_pictures_are_capped_at_the_viewers_largest(capsys, images):
    init(capsys, images)  # viewer claude (default): up to 2000 px
    code, envelope = cli(capsys, str(images), "look", "--mode", "section", "--sections", ID1,
                         "--resolution", "5000")
    assert code == 0, envelope
    assert "used 2000" in envelope["result"]["resolution_note"]
    init(capsys, images, "--viewer", "codex")
    code, envelope = cli(capsys, str(images), "look", "--mode", "section", "--sections", ID1,
                         "--resolution", "5000")
    assert "used 2048" in envelope["result"]["resolution_note"]


def test_an_image_model_not_connected_is_reported_and_its_verb_refused(
        capsys, images, monkeypatch):
    init(capsys, images, "--image-provider", "openai-oauth")  # HOME is private: no login
    code, status = cli(capsys, str(images), "status")
    assert code == 0
    assert status["result"]["image_model"] == {
        "provider": "openai-oauth", "connected": False, "trace_borders": False}
    assert "trace_borders" not in status["result"]["verbs"]
    code, refused = cli(capsys, str(images), "trace_borders", "--id", ID0)
    assert code == 3 and refused["error"]["code"] == "IMAGE_MODEL_OFF"
    assert "langslice login" in refused["error"]["fix"]
    code, brief = cli(capsys, str(images), "brief")
    from langslice.doors.statement import IMAGE_MODEL_OFF

    assert IMAGE_MODEL_OFF in brief["result"]["statement"]
    # Connected (a login present), the verb is offered, as the MCP door offers it.
    from langslice.doors.api import setup

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
    # Every row carries both fields (uniform rows): set where set, else empty.
    assert rows[ID0]["frozen"] == ["orientation", "affine"]
    assert rows[ID0]["damage_by_user"] is False
    assert rows[ID1]["damage_by_user"] is True and rows[ID1]["frozen"] == []
    assert rows[ID2]["frozen"] == [] and rows[ID2]["damage_by_user"] is False
    # The user's damage note is a note: damaged regions make a section damaged.
    assert rows[ID1]["damage_note"] == "torn" and rows[ID1]["damaged"] is False
    assert rows[ID1]["damaged_regions"] == []


def test_every_call_is_logged_and_traced(capsys, images, monkeypatch, tmp_path):
    traces = tmp_path / "traces"
    monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(traces))
    job = init(capsys, images)
    code, envelope = cli(capsys, str(images), "look", "--mode", "section", "--sections", ID0)
    assert code == 0
    code, refused = cli(capsys, str(images), "undo")
    assert code == 3
    lines = [json.loads(line) for line in (job / "logs" / "calls.jsonl").read_text()
             .splitlines()]
    assert [line["verb"] for line in lines] == ["init", "look", "undo"]
    seen = lines[1]
    assert seen["arguments"] == ["--mode", "section", "--sections", ID0] and seen["ok"] is True
    assert seen["exit"] == 0 and seen["artifacts"] == [
        item["path"] for item in envelope["artifacts"]]
    assert lines[2]["ok"] is False and lines[2]["error"] == "NOTHING_TO_UNDO"
    # Each call's whole envelope, as printed, is kept beside the log; the line names it.
    assert [line["reply"] for line in lines] == [
        "logs/calls/000001_init.json", "logs/calls/000002_look.json",
        "logs/calls/000003_undo.json"]
    assert json.loads((job / lines[1]["reply"]).read_text()) == envelope
    assert json.loads((job / lines[2]["reply"]).read_text()) == refused
    assert lines[2]["message"] == refused["error"]["message"]
    # The trace, as the MCP door writes its own: one file for the job.
    files = list(traces.glob("cli_stack_*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [record["kind"] for record in records] == ["tool_result"] * 3
    traced = records[1]
    assert traced["name"] == "look" and traced["args"]["sections"] == [ID0]
    assert json.loads(traced["content"][0]["text"])["ok"] is True
    assert traced["content"][1]["image"] == "image/jpeg" and traced["content"][1]["bytes"] > 0
    # The artifacts: index and label for each picture, in the order the
    # reply's `pictures` lists their numbers.
    view = envelope["artifacts"][0]
    assert view["kind"] == "view" and view["index"] == 0 and ID0 in view["label"]
    picture, = envelope["result"]["pictures"]
    assert ID0 in picture["caption"] and Path(view["path"]).parent.name.startswith(
        f"{picture['id']:06d}_")


# --- writes, pictures, exit codes -----------------------------------------------------


def test_a_write_returns_its_pictures_as_artifact_paths(capsys, images):
    init(capsys, images)
    code, envelope = cli(capsys, str(images), "position-sections", "--args",
                         json.dumps({"sections": [{"id": ID0, "position_mm": 0.1}]}))
    assert code == 0, envelope
    assert envelope["result"]["written"] == [{"id": ID0, "position_mm": 0.1}]
    assert "description" not in envelope["result"] and "images" not in envelope["result"]
    # The positioning picture of the written section, drawn after the write.
    kinds = [artifact["kind"] for artifact in envelope["artifacts"]]
    assert kinds == ["view", "view_json"]
    for artifact in envelope["artifacts"]:
        assert Path(artifact["path"]).is_file() and os.path.isabs(artifact["path"])
    record = json.loads(Path(envelope["artifacts"][1]["path"]).read_text())
    assert record["tool"] == "position_sections" and record["sections"] == [ID0]
    assert record["mode"] == "positioning"
    assert len(envelope["result"]["pictures"]) == 1
    # A section under its new transform: the overlay, with its layers.
    code, turned = cli(capsys, str(images), "interactive_transform", "--sections",
                       json.dumps([{"id": ID0, "rotation_deg": 2.0}]))
    assert code == 0, turned
    assert [artifact["kind"] for artifact in turned["artifacts"]] == [
        "view", "view_json", "labels", "borders"]
    record = json.loads(Path(turned["artifacts"][1]["path"]).read_text())
    assert record["tool"] == "interactive_transform" and record["mode"] == "overlay"
    # The pictures to open come first, before the reply: their JPEGs' paths.
    assert list(turned)[:3] == ["ok", "picture_files", "result"]
    assert turned["picture_files"] == [turned["artifacts"][0]["path"]]
    # view false: no picture.
    code, blind = cli(capsys, str(images), "interactive_transform", "--sections",
                      json.dumps([{"id": ID0, "rotation_deg": 3.0}]), "--view", "false")
    assert code == 0 and blind["artifacts"] == [] and "pictures" not in blind["result"]
    assert "picture_files" not in blind
    # A number names no section: refused with the filenames.
    code, numbered = cli(capsys, str(images), "position_sections", "--sections",
                         '[{"id": 1, "position_mm": 0.15}]')
    assert code != 0 and ID1 in numbered["result"]["filenames"]
    assert "filename" in numbered["result"]["message"]
    # The flag form, the filename without its extension, the verbose reply.
    code, verbose = cli(capsys, str(images), "position_sections", "--sections",
                        json.dumps([{"id": ID1.rsplit(".", 1)[0], "position_mm": 0.15}]),
                        "--verbose")
    assert code == 0 and verbose["result"]["written"] == [{"id": ID1, "position_mm": 0.15}]


def test_exit_codes_and_fix_hints(capsys, images):
    init(capsys, images)
    code, envelope = cli(capsys, str(images), "position_sections", "--sections", "[]",
                         "--bogus", "1")
    assert code == 2 and envelope["ok"] is False
    assert envelope["error"]["code"] == "UNKNOWN_ARGUMENTS"
    job = images / "langslice"
    fix = envelope["error"]["fix"]
    assert fix.startswith("position_sections takes sections (") and "cutting_angles (" in fix
    assert (f"Call it as: `langslice-job {job} position_sections --sections '[]' "
            "--cutting-angles '{}' --view true`.") in fix
    # The guessed names of the file tools: what each argument means, and the
    # call without the unknown ones (no aliases are accepted).
    code, envelope = cli(capsys, str(images), "read_file", "--path", "AGENTS.md",
                         "--start", "10", "--end", "40")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_ARGUMENTS"
    assert "accepted: path, offset, limit" in envelope["error"]["message"]
    assert envelope["error"]["fix"] == (
        "read_file takes path (The file, relative to the job folder; required); offset (Lines "
        "to skip from the start; default 0); limit (Lines to show, at most 2000; default 400). "
        f"Call it as: `langslice-job {job} read_file --path AGENTS.md --offset 0 --limit "
        "400`.")
    code, envelope = cli(capsys, str(images), "list_files", "--recursive", "--limit", "5")
    assert code == 2 and "glob (Every file below path whose name or path matches this glob " \
        "(\"*.json\", \"views/*\"; \"*\" is every file below path); default \"\")" \
        in envelope["error"]["fix"]
    assert f"`langslice-job {job} list_files --path . --glob ''`" in envelope["error"]["fix"]
    code, envelope = cli(capsys, str(images), "read_file", "--path", "sections/nope/x.json")
    assert code == 2 and envelope["error"]["code"] == "NOT_FOUND"
    assert "the nearest folder above it that does, 'sections', holds:" in (
        envelope["error"]["message"])
    assert "nearest folder above the path" in envelope["error"]["fix"]
    code, envelope = cli(capsys, str(images), "look")
    assert code == 2 and envelope["error"]["code"] == "MISSING_ARGUMENTS"
    assert envelope["result"] == {"missing": ["mode"]}
    code, envelope = cli(capsys, str(images), "look", "--mode", "section",
                         "--sections", "nope.png")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_SLICE_IDS"
    code, envelope = cli(capsys, str(images), "position_sections", "--args", "{not json")
    assert code == 2 and envelope["error"]["code"] == "BAD_JSON"
    code, envelope = cli(capsys, str(images), "undo")
    assert code == 3 and envelope["error"]["code"] == "NOTHING_TO_UNDO"
    code, envelope = cli(capsys, str(images), "submit", "--summary", "done",
                         "--notes", "[]", "--interval-breaks", "[]")
    assert code == 3 and envelope["error"]["code"] == "MISSING_POSITIONS"
    assert envelope["error"]["fix"] and envelope["result"]["missing_ids"] == [ID0, ID1, ID2]
    # No image model in this job: no trace_borders.
    code, envelope = cli(capsys, str(images), "trace_borders", "--section", ID0)
    assert code == 3 and envelope["error"]["code"] == "VERB_OFF"
    code, envelope = cli(capsys, str(images), "fit_affine")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_VERB"


def test_an_internal_error_exits_4(capsys, images, monkeypatch):
    init(capsys, images)
    import langslice.ops.notes as notes

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(notes, "add_note", broken)
    code, envelope = cli(capsys, str(images), "note", "--text", "hello")
    assert code == 4 and envelope["error"]["code"] == "INTERNAL"
    assert "boom" in envelope["error"]["message"]


def test_a_first_sentence_ends_at_a_sentence_not_an_abbreviation():
    from langslice.doors.cli.job import _first_sentence

    assert _first_sentence('A glob (e.g. "*.json"): every file. Empty lists it.') == (
        'A glob (e.g. "*.json"): every file.')
    assert _first_sentence("Steps, e.g. three. More.") == "Steps, e.g. three."
    assert _first_sentence("No full stop") == "No full stop"


def test_folder_ops_lists_the_job_s_own_verbs(capsys, images):
    job = init(capsys, images, "--tasks", "position")
    code, envelope = cli(capsys, str(images), "ops")
    assert code == 0, envelope
    names = [row["name"] for row in envelope["result"]["verbs"]]
    assert envelope["result"]["job"] == str(job)
    assert "look" in names and "position_sections" in names
    assert "fireants_syn" not in names and "elastix_affine" not in names
    assert envelope["next"][0] == f"langslice-job {job} brief"
    code, envelope = cli(capsys, str(images), "ops", "--bogus")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS"


def test_a_failed_fit_says_the_first_line_and_keeps_the_rest_in_the_log(tmp_path):
    from langslice.core.engine_output import failure_message

    log = tmp_path / "logs" / "engine.log"
    text = ("AcceleratorError: CUDA error: out of memory\nSearch for "
            "`cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/...\nmore lines")
    words = failure_message(text, log, "the fit of s0.png failed")
    assert words.startswith("AcceleratorError: CUDA error: out of memory. The GPU ran out of "
                            "memory")
    assert "cudaErrorMemoryAllocation" not in words
    assert words.endswith("logs/engine.log holds the whole error.")
    held = log.read_text()
    assert "the fit of s0.png failed" in held and "more lines" in held
    assert failure_message("ValueError: bad", None) == "ValueError: bad."


def test_dry_run_reports_the_change_and_writes_nothing(capsys, images):
    job = init(capsys, images)
    state_before = (job / "state.json").read_bytes()

    def files() -> list[str]:  # every call's reply is kept in logs/calls/, a dry run's too
        return sorted(path.name for path in job.rglob("*") if "calls" not in path.parts)

    views_before = files()
    code, envelope = cli(capsys, str(images), "position_sections", "--sections",
                         json.dumps([{"id": ID1, "position_mm": 0.15}]), "--dry-run")
    assert code == 0, envelope
    assert envelope["result"]["dry_run"] is True
    assert envelope["result"]["would_change"] == {
        "sections": {ID1: ["position_mm"]}, "stack": []}
    assert envelope["artifacts"] == []
    assert envelope["next"] and "--dry-run" not in envelope["next"][0]
    assert (job / "state.json").read_bytes() == state_before
    assert files() == views_before
    # A fit is checked, not run.
    for fit in ("fireants_syn", "elastix_affine"):
        code, envelope = cli(capsys, str(images), fit, "--sections", ID0, "--dry-run")
        assert code == 0 and envelope["result"] == {"dry_run": True, "simulated": False,
                                                    "sections": [ID0]}, fit
        # The real call is safe in the foreground (it answers within 30 s).
        assert envelope["next"] == [f"langslice-job {job} {fit} --args "
                                    f"'{json.dumps({'sections': [ID0]})}'"]
    code, envelope = cli(capsys, str(images), "fireants_syn", "--sections", "nope.png",
                         "--dry-run")
    assert code == 2 and envelope["error"]["code"] == "UNKNOWN_SLICE_IDS"
    # A dry run never starts a background run (that child would run the verb).
    code, envelope = cli(capsys, str(images), "fireants_syn", "--sections", ID0,
                         "--dry-run", "--background")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS"
    assert not (job / "logs" / "runs").exists()
    assert (job / "state.json").read_bytes() == state_before


def test_export_maps_writes_the_derived_files_as_artifacts(capsys, images):
    job = init(capsys, images)
    code, _placed = cli(capsys, str(images), "position_sections", "--sections", json.dumps(
        [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
         {"id": ID2, "position_mm": 0.2}]), "--view", "false")
    assert code == 0, _placed
    code, envelope = cli(capsys, str(images), "export-maps", "--sections", ID0,
                         "--dry-run")
    assert code == 0 and envelope["result"]["files_written"] is False
    assert envelope["artifacts"] == [] and not (job / "sections/s0/coords.tif").exists()
    code, envelope = cli(capsys, str(images), "export_maps", "--sections", ID0)
    assert code == 0, envelope
    assert envelope["result"]["written"] == [ID0] and envelope["result"]["n_files"] >= 5
    kinds = {artifact["kind"] for artifact in envelope["artifacts"]}
    assert {"coords", "labels", "labels_fiji", "labels_csv", "maps", "quicknii",
            "visualign", "registration"} <= kinds
    assert all(Path(artifact["path"]).is_file() for artifact in envelope["artifacts"])
    # The verb is the CLI's and the library's: listed by status, never a model's tool.
    code, envelope = cli(capsys, str(images), "status")
    assert "export_maps" in envelope["result"]["verbs"]


def test_trace_borders_answers_once_its_background_work_has_landed(capsys, images,
                                                                   monkeypatch):
    """The packaged trace starts background work; the CLI process waits for
    it and answers with its notice and its pictures as artifacts."""
    pytest.importorskip("fireants")
    from langslice.doors.api import setup
    from langslice.doors.tools import toolbox
    from tests.synthetic_stack import stub_image_model

    monkeypatch.setattr(setup, "image_model_connected", lambda provider: True)
    monkeypatch.setattr(toolbox, "resolve_image_model", lambda provider, model: stub_image_model(
        SimpleNamespace(nonlinear=SimpleNamespace(provider=provider, image_model=model))))
    job = init(capsys, images, "--image-provider", "openai-oauth")
    code, placed = cli(capsys, str(images), "position_sections", "--sections",
                       json.dumps([{"id": ID0, "position_mm": 0.05}]), "--view", "false")
    assert code == 0, placed
    code, envelope = cli(capsys, str(images), "interactive_transform", "--sections",
                         json.dumps([{"id": ID0, "rotation_deg": 0.0}]), "--view", "false")
    assert code == 0, envelope
    code, traced = cli(capsys, str(images), "trace_borders", "--section", ID0)
    assert code == 0, traced
    result = traced["result"]
    assert result["status"] == "started" and result["id"] == ID0
    work = result["work"]
    # Landed in this process: its notice opens the reply, its result is attached.
    assert result["work_status"] == "done", result
    assert result["background"] and work in json.dumps(result["background"])
    worked = [entry for entry in result.get("pictures", []) if entry.get("work") == work]
    pictures = [item for item in traced["artifacts"] if item["kind"] == "view"
                and item["label"].startswith(f"{work} (")]
    assert worked and len(pictures) == len(worked)
    assert all(Path(item["path"]).is_file() for item in pictures)
    state = json.loads((job / "state.json").read_text())
    section = next(row for row in state["slices"] if row["id"] == ID0)
    assert section["deformation"]


# --- background runs --------------------------------------------------------------------


def test_a_background_run_answers_at_once_then_status_and_wait(capsys, images, monkeypatch):
    from langslice.doors.cli import background

    init(capsys, images)
    # install() pointed background runs at tests.cli_child (the synthetic atlas).
    assert background.CHILD_COMMAND[-1] == "tests.cli_child"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO / "src"), str(REPO))))
    monkeypatch.chdir(REPO)
    code, envelope = cli(capsys, str(images), "look", "--mode", "section", "--sections", ID0,
                         "--background")
    assert code == 0 and envelope["result"]["state"] == "running"
    run = envelope["result"]["run"]
    assert envelope["next"] == [f"langslice-job {images / 'langslice'} wait {run}"]
    code, waited = cli(capsys, str(images), "wait", run, "--timeout", "120")
    assert code == 0, waited
    assert waited["result"]["run"] == run
    picture, = waited["result"]["pictures"]
    assert picture["caption"].startswith(f"{ID0} section")
    assert [artifact["kind"] for artifact in waited["artifacts"]] == ["view", "view_json"]
    assert Path(waited["artifacts"][0]["path"]).is_file()
    code, one = cli(capsys, str(images), "runs", run)
    assert code == 0 and one["result"]["state"] == "finished"
    assert one["result"]["exit"] == 0
    code, listed = cli(capsys, str(images), "runs")
    assert listed["result"]["runs"][0]["id"] == run
    code, unknown = cli(capsys, str(images), "runs", "no-such-run")
    assert code == 2 and unknown["error"]["code"] == "UNKNOWN_RUN"
    # `status` is only the verb.
    code, status = cli(capsys, str(images), "status", run)
    assert code == 2 and status["error"]["code"] == "BAD_ARGUMENTS"


def test_wait_answers_still_running_within_its_default_wait(capsys, images, monkeypatch):
    """A harness ends a command it finds too long, so `wait` answers by itself
    (``background.WAIT_S``, 30 s) with the run still running and how to wait
    again: an answer (ok, exit 0), not an error."""
    import time

    from langslice.doors.cli import background
    from langslice.job.layout import JobLayout

    job = init(capsys, images)
    layout = JobLayout(job)
    run_id = "20261021-000000-abcdef"
    background._write(layout, {"id": run_id, "verb": "fireants_syn", "arguments": {},
                               "state": "running", "started_at": "2026-10-21T00:00:00+00:00",
                               "started": time.time(), "pid": os.getpid()})  # alive: this one
    monkeypatch.setattr(background, "WAIT_S", 0.5)
    started = time.monotonic()
    code, envelope = cli(capsys, str(images), "wait", run_id)
    assert time.monotonic() - started < 10
    assert code == 0 and envelope["ok"] is True and "error" not in envelope
    assert envelope["result"]["state"] == "running" and envelope["result"]["id"] == run_id
    assert envelope["result"]["waited_s"] == 0.5
    assert "Still running after 0.5 s" in envelope["result"]["message"]
    assert envelope["next"] == [f"langslice-job {job} wait {run_id}"]
    # --timeout still sets it.
    started = time.monotonic()
    code, envelope = cli(capsys, str(images), "wait", run_id, "--timeout", "1.5")
    assert code == 0 and time.monotonic() - started >= 1.5
    assert envelope["result"]["waited_s"] == 1.5
    # A run whose process is gone is still a failure.
    background._write(layout, {"id": run_id, "verb": "fireants_syn", "arguments": {},
                               "state": "running", "started_at": "2026-10-21T00:00:00+00:00",
                               "started": time.time(), "pid": 2 ** 22 + 12345})
    code, lost = cli(capsys, str(images), "wait", run_id)
    assert code == 4 and lost["ok"] is False and lost["error"]["code"] == "RUN_LOST"


def test_detaching_follows_the_setting_and_claude_code(monkeypatch):
    """A long verb's foreground call runs detached unless LANGSLICE_DETACH says
    no, or, unset, under Claude Code, whose sandbox ends what a command leaves."""
    from langslice.doors.cli.background import DETACH_ENV, detaching

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
    from langslice.doors.cli import background
    from langslice.job.layout import JobLayout

    init(capsys, images)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO / "src"), str(REPO))))
    monkeypatch.chdir(REPO)
    code, placed = cli(capsys, str(images), "position_sections", "--args", json.dumps(
        {"sections": [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15}],
         "view": False}))
    assert code == 0, placed
    monkeypatch.setenv(background.DETACH_ENV, "1")
    monkeypatch.setattr(background, "WAIT_S", 120.0)
    code, fitted = cli(capsys, str(images), "elastix_affine", "--sections", ID0)
    assert code == 0, fitted
    run = fitted["result"]["run"]
    (row,) = fitted["result"]["results"]
    assert row["id"] == ID0 and row["status"] == "ok"
    assert fitted["picture_files"] and Path(fitted["picture_files"][0]).is_file()
    assert background.read(JobLayout(images / "langslice"), run)["state"] == "finished"
    # A wait shorter than the fit: the call answers with the run still going.
    monkeypatch.setattr(background, "WAIT_S", 0.01)
    code, running = cli(capsys, str(images), "elastix_affine", "--sections", ID1)
    assert code == 0 and running["ok"] is True, running
    assert running["result"]["state"] == "running"
    later = running["result"]["id"]
    assert running["next"] == [f"langslice-job {images / 'langslice'} wait {later}"]
    code, landed = cli(capsys, str(images), "wait", later, "--timeout", "120")
    assert code == 0 and landed["result"]["run"] == later
    assert landed["result"]["results"][0]["id"] == ID1


# --- live shared editing --------------------------------------------------------------


def test_cli_calls_and_a_running_toolbox_interleave_on_one_folder(capsys, images):
    from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec
    from langslice.doors.jobs import context
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import Job

    job_folder = init(capsys, images)
    code, envelope = cli(capsys, str(images), "status")
    start = {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]}
    # The agent's side: the job folder opened as a run opens it, its tools.
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

    # The CLI sees the agent's write and adds its own.
    code, envelope = cli(capsys, str(images), "position_sections", "--sections",
                         json.dumps([{"id": ID1, "position_mm": 0.15}]))
    assert code == 0
    code, envelope = cli(capsys, str(images), "status")
    positions = {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]}
    assert positions == {ID0: 0.1, ID1: 0.15, ID2: start[ID2]}

    # The agent's next call picks up the CLI's write, history included.
    rows = tools["status"]()["rows"]
    assert {row["id"]: row.get("position_mm") for row in rows} == positions
    assert tools["undo"]()["status"] == "ok"  # undoes the CLI's write
    assert agent.state.resolve(ID1).position_mm == start[ID1]
    assert agent.state.resolve(ID0).position_mm == 0.1

    # And the CLI sees the agent's undo; its own undo takes the agent's write back.
    code, envelope = cli(capsys, str(images), "undo")
    assert code == 0
    code, envelope = cli(capsys, str(images), "status")
    assert {row["id"]: row.get("position_mm") for row in envelope["result"]["rows"]} == start
    assert tools["status"]()["rows"][0].get("position_mm") == start[ID0]
    agent.close()


# --- the library ------------------------------------------------------------------------


def test_open_job_gives_the_verbs_as_methods(capsys, images):
    import langslice

    init(capsys, images)
    with langslice.open_job(images) as job:
        assert "position_sections" in job.verbs and "position_sections" in dir(job)
        reply = job.position_sections(sections=[{"id": ID2, "position_mm": 0.2}])
        # A plain JSON reply; the pictures as files already on disk (as the
        # CLI lists them), the images themselves on an attribute.
        assert reply["status"] == "ok" and "images" not in reply
        assert json.loads(json.dumps(reply)) == reply
        views = [item for item in reply["artifacts"] if item["kind"] == "view"]
        assert views and all(Path(item["path"]).is_file() for item in reply["artifacts"])
        assert [item["index"] for item in views] == list(range(len(reply.images)))
        assert job.state.resolve(ID2).position_mm == 0.2
        # Every status row carries every field: the last placed section's
        # delta_to_next_mm is null, not missing.
        rows = job.status()["rows"]
        assert len({tuple(row) for row in rows}) == 1
        assert rows[-1]["delta_to_next_mm"] is None
        # A verb this job's settings have not: said so.
        with pytest.raises(AttributeError, match="This job's settings have no 'trace_borders'"):
            getattr(job, "trace_borders")  # noqa: B009
        # A section on its atlas: its picture's frame gives atlas coordinates.
        overlay = job.look(mode="overlay", sections=[ID2])
        saved = next(item["path"] for item in overlay["artifacts"]
                     if item["kind"] == "view_json")
    code, envelope = cli(capsys, str(images), "status")
    assert envelope["result"]["rows"][2]["position_mm"] == 0.2
    xyz = langslice.coordinate_map(saved)
    assert xyz.ndim == 3 and xyz.shape[2] == 3 and bool((xyz == xyz).any())


# --- a dead background run on Windows ------------------


def _dead_run(tmp_path: Path) -> tuple[Any, str]:
    from langslice.doors.cli import background
    from langslice.job.layout import JobLayout

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

    from langslice.doors.cli import background

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


def test_the_cli_and_the_library_cap_no_pictures_or_sections_but_fireants_syn(capsys, images):
    """``Verb.limits`` (what the card lists) holds the compute cap only; the
    pictures and the transform sections of one call are not capped."""
    import langslice
    from langslice.ops.registry import VERBS

    init(capsys, images)
    limits = {name: dict(verb.limits) for name, verb in VERBS.items() if verb.limits}
    assert limits == {"fireants_syn": {"sections": limits["fireants_syn"]["sections"]}}
    with langslice.open_job(images) as job:
        code, _placed = cli(capsys, str(images), "position_sections", "--args", json.dumps(
            {"sections": [{"id": ID0, "position_mm": 0.1}, {"id": ID1, "position_mm": 0.15},
                          {"id": ID2, "position_mm": 0.2}], "view": False}))
        assert code == 0, _placed
        most = limits["fireants_syn"]["sections"]
        refused = job.fireants_syn(sections=[ID0] * (most + 1))
        assert refused["error"] == "TOO_MANY_SECTIONS" and refused["max_sections"] == most
        # The refusal says why the cap holds.
        assert most == 20 and refused["message"] == (
            "fireants_syn takes at most 20 sections per call (every fitted section is held in "
            "memory (about 170 MB) until the call writes them all as one undo step); 21 were "
            "named. Nothing was done: split them over calls.")
        # Past what the ADK door takes: every section written, every picture shown.
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
    from types import SimpleNamespace

    from langslice.ops.refusal import Refused
    from langslice.ops.registry import check_section_count

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


# --- a script that exits without closing its job ----------------------------------------

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
    lines = [json.loads(line) for line in (job / "views.jsonl").read_text().splitlines()]
    placed = [line for line in lines if line["tool"] == "interactive_transform"]
    assert len(placed) == 3 and all(line["layers"] for line in placed)
    for line in placed:
        assert (job / line["path"] / "labels.tif").is_file()


# --- tools and options a job disables ----------------------------------------------------


def test_disable_takes_tools_and_options_off_a_cli_job(capsys, images):
    """``--disable``: the verb is not offered, an option is not in the schema,
    a call of it is refused, and nothing the agent reads names either."""
    code, envelope = cli(capsys, str(images), "init", "--tasks", "position,transform",
                         "--image-provider", "none", "--preprocess", "none",
                         "--pixel-size-um", str(PIXEL_SIZE_UM), "--disable", "zoom",
                         "--disable", "look:positioning,interactive_transform:flip")
    assert code == 0, envelope
    folder = Path(envelope["result"]["job_folder"])
    assert "zoom" not in envelope["result"]["verbs"] and "look" in envelope["result"]["verbs"]
    assert json.loads((folder / "job.json").read_text())["spec"]["disabled"] == [
        "zoom", "look:positioning", "interactive_transform:flip"]
    statement = envelope["result"]["statement"]
    assert "`zoom`" not in statement and "mode positioning" not in statement
    card = (folder / "AGENTS.md").read_text()
    assert "- `zoom`" not in card and "- `look`" in card
    assert (folder / "CLAUDE.md").read_text() == card

    code, refused = cli(capsys, str(folder), "zoom", "--box", "[0,0,1,1]")
    assert code == 3 and refused["error"]["code"] == "VERB_OFF"
    assert "disabled" in refused["error"]["message"]
    code, refused = cli(capsys, str(folder), "look", "--mode", "positioning")
    assert code == 3 and refused["error"]["code"] == "OPTION_OFF"
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
    code, envelope = cli(capsys, str(images), "init", "--disable", "zoomm")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS"
    assert "unknown name 'zoomm'" in envelope["error"]["message"]
    assert "look:positioning" in envelope["error"]["message"]
    code, envelope = cli(capsys, str(images), "init", "--tasks", "transform",
                         "--disable", "elastix_affine,interactive_transform")
    assert code == 2 and "every tool that writes its answer" in envelope["error"]["message"]
