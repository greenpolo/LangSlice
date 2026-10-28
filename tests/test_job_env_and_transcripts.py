"""The agent CLI's ``env`` (the job's own Python environment) and the
harness plugins' ``langslice session-end`` hook (a coding agent's transcript
copied into its job)."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from langslice.doors.cli import jobenv, transcripts
from langslice.doors.cli.jobcli import main
from langslice.job.layout import JobLayout
from tests.cli_child import install
from tests.synthetic_stack import PIXEL_SIZE_UM, atlas_loader, write_sections


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("env") / "stack"
    write_sections(images)
    return images


@pytest.fixture
def job(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    install(atlas_loader(), monkeypatch.setattr)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    code, envelope = cli(capsys, str(folder), "init", "--tasks", "position,transform",
                         "--image-provider", "none", "--preprocess", "none",
                         "--pixel-size-um", str(PIXEL_SIZE_UM))
    assert code == 0, envelope
    return Path(envelope["result"]["job_folder"])


def cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = main(list(argv))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    return int(code or 0), json.loads(lines[0])


def wheel(folder: Path, name: str = "lsdemo", version: str = "0.1") -> Path:
    """A minimal pure-Python wheel (installs offline)."""
    path = folder / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": "VALUE = 42\n",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                         "Tag: py3-none-any\n",
    }
    record = "".join(f"{key},,\n" for key in files) + f"{info}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as archive:
        for key, text in {**files, f"{info}/RECORD": record}.items():
            archive.writestr(key, text)
    return path


# --- env -------------------------------------------------------------------------------


def test_env_imports_langslice_installs_lists_and_rebuilds(capsys, job, tmp_path):
    code, envelope = cli(capsys, str(job), "env")
    assert code == 0, envelope
    python = Path(envelope["result"]["python"])
    assert python.parent.parent == job / "env" and envelope["result"]["packages"] == []
    done = subprocess.run([str(python), "-c", "import langslice, numpy; print('ok')"],
                          capture_output=True, text=True, check=False)
    assert done.stdout.strip() == "ok", done.stderr

    code, envelope = cli(capsys, str(job), "env", "install", str(wheel(tmp_path)))
    assert code == 0, envelope
    assert envelope["result"]["packages"] == ["lsdemo==0.1"]
    assert (job / "scripts" / "requirements.txt").read_text() == "lsdemo==0.1\n"
    assert subprocess.run([str(python), "-c", "import lsdemo; assert lsdemo.VALUE == 42"],
                          check=False).returncode == 0
    assert not list((Path(sys.prefix) / "lib").glob("python*/site-packages/lsdemo"))

    code, envelope = cli(capsys, str(job), "env", "remove")
    assert code == 0 and envelope["result"]["removed"] is True
    assert not (job / "env").exists()
    assert (job / "scripts" / "requirements.txt").is_file()


def test_env_refuses_other_arguments(capsys, job):
    code, envelope = cli(capsys, str(job), "env", "upgrade")
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS"
    code, envelope = cli(capsys, str(job), "env", "install")
    assert code == 2


def test_env_of_a_folder_without_a_job(capsys, tmp_path):
    code, envelope = cli(capsys, str(tmp_path), "env")
    assert envelope["error"]["code"] == "NO_JOB" and code != 0


# --- session-end ----------------------------------------------------------------------


def _lines(*records: dict[str, Any]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def _claude_transcript(job: Path, *, submit: bool) -> list[dict[str, Any]]:
    picture = base64.b64encode(b"x" * 30000).decode()
    records = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": f'langslice-job "{job}" status'}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": picture}}]}]}},
    ]
    if submit:
        records.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": f"langslice-job {job} submit"}}]}})
    return records


def _run_hook(hook: dict[str, Any], harness: str, monkeypatch: pytest.MonkeyPatch,
              capsys: pytest.CaptureFixture[str]) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(hook)))
    transcripts.run(type("Args", (), {"harness": harness})())
    return capsys.readouterr().out.strip()


def test_a_claude_code_subagent_transcript_lands_in_its_job(capsys, job, tmp_path,
                                                            monkeypatch):
    jobenv.ensure(JobLayout(job))
    session = tmp_path / "projects" / "sess1"
    (session / "subagents").mkdir(parents=True)
    (session / "subagents" / "agent-abc.jsonl").write_text(
        _lines(*_claude_transcript(job, submit=True)))
    hook = {"session_id": "sess1", "transcript_path": f"{session}.jsonl",
            "agent_id": "abc", "agent_type": "langslice:register",
            "cwd": str(tmp_path), "hook_event_name": "SubagentStop"}
    assert _run_hook(hook, "claude-code", monkeypatch, capsys) == "{}"
    copied = (job / "trace" / "claude-code_abc.jsonl").read_text()
    assert "<30000 bytes of inline data>" in copied
    assert "x" * 1000 not in copied and "langslice-job" in copied
    assert not (job / "env").exists()  # the session submitted: the environment is pruned


def test_an_antigravity_session_that_did_not_submit_keeps_the_environment(
        capsys, job, tmp_path, monkeypatch):
    jobenv.ensure(JobLayout(job))
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(_lines(
        {"step_index": 0, "type": "USER_INPUT", "content": "Register the brain"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "run_command", "args": {"CommandLine": f"langslice-job {job} status"}}]}))
    hook = {"conversationId": "conv-1", "transcriptPath": str(transcript),
            "workspacePaths": [str(job)], "terminationReason": "NO_TOOL_CALL"}
    assert _run_hook(hook, "antigravity", monkeypatch, capsys) == "{}"
    assert (job / "trace" / "antigravity_conv-1.jsonl").is_file()
    assert (job / "env").is_dir()


@pytest.mark.parametrize("harness,records", [
    ("claude-code", lambda image_folder, job_folder: [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__plugin_langslice_langslice__start_job",
             "input": {"image_folder": image_folder}}]}}]),
    ("codex", lambda image_folder, job_folder: [
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "mcp__langslice__start_job",
            "arguments": json.dumps({"image_folder": job_folder})}}]),
    ("antigravity", lambda image_folder, job_folder: [
        {"step_index": 1, "type": "PLANNER_RESPONSE", "tool_calls": [
            {"name": "call_mcp_tool", "args": {
                "ServerName": "langslice", "ToolName": "start_job",
                "Arguments": json.dumps({"image_folder": image_folder})}}]}]),
], ids=["claude-code", "codex", "antigravity"])
def test_an_mcp_agent_transcript_lands_in_the_job_its_start_job_opened(
        capsys, job, tmp_path, monkeypatch, harness, records):
    """An MCP registration agent names its job only in start_job's arguments."""
    transcript = tmp_path / "mcp.jsonl"
    transcript.write_text(_lines(*records(str(job.parent), str(job))))
    found = transcripts.job_folders(transcripts._said(transcript.read_text()), tmp_path,
                                    also=transcripts.started_jobs(transcript.read_text()))
    assert found == [job]
    if harness != "codex":  # the session-end hook is Claude Code's and Antigravity's
        hook = {"conversationId": "c-mcp", "transcriptPath": str(transcript)}
        assert _run_hook(hook, harness, monkeypatch, capsys) == "{}"
        assert (job / "trace" / f"{harness}_c-mcp.jsonl").is_file()


@pytest.mark.parametrize("submit", [
    {"type": "tool_use", "name": "mcp__plugin_langslice_langslice__submit", "input": {}},
    {"name": "call_mcp_tool", "args": {"ServerName": "langslice_mcp", "ToolName": "submit"}},
], ids=["claude-code", "antigravity"])
def test_an_mcp_submit_removes_the_environment(capsys, job, tmp_path, monkeypatch, submit):
    """A job first worked with scripts, then submitted through the MCP tools."""
    jobenv.ensure(JobLayout(job))
    transcript = tmp_path / "mcp_submit.jsonl"
    transcript.write_text(_lines(
        {"type": "tool_use", "name": "mcp__langslice__start_job",
         "input": {"image_folder": str(job)}}, {"calls": [submit]}))
    assert transcripts.called(transcript.read_text(), transcripts.MCP_SUBMIT)
    hook = {"conversationId": "c-sub", "transcriptPath": str(transcript)}
    assert _run_hook(hook, "claude-code", monkeypatch, capsys) == "{}"
    assert not (job / "env").exists()


def test_a_start_job_by_saved_job_id_lands_in_that_job(capsys, job, tmp_path, monkeypatch):
    from langslice.job import index

    index.register(index.default_root(), "a1b2c3d4e5f6", job)
    transcript = tmp_path / "saved.jsonl"
    transcript.write_text(_lines({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "mcp__langslice__start_job",
         "input": {"job_id": "a1b2c3d4e5f6"}}]}}))
    assert transcripts.started_jobs(transcript.read_text()) == [str(job)]
    # A start_job of an unknown id or with no arguments names nothing.
    assert transcripts.started_jobs(_lines(
        {"name": "mcp__langslice__start_job", "input": {"job_id": "no-such-job"}},
        {"name": "mcp__langslice__start_job", "input": {}})) == []


def test_a_transcript_naming_no_job_copies_nothing_and_never_fails(
        capsys, tmp_path, monkeypatch):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("not json\n" + _lines({"command": "langslice-job ops"}))
    hook = {"transcriptPath": str(transcript), "conversationId": "c"}
    assert _run_hook(hook, "antigravity", monkeypatch, capsys) == "{}"
    assert _run_hook({"transcriptPath": str(tmp_path / "missing.jsonl")}, "antigravity",
                     monkeypatch, capsys) == "{}"
    monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))
    transcripts.run(type("Args", (), {"harness": "antigravity"})())
    assert capsys.readouterr().out.strip() == "{}"


def test_slim_keeps_text_and_replaces_inline_data():
    picture = base64.b64encode(b"\x00" * 9000).decode()
    slimmed = transcripts.slim({"text": "a" * 10, "data": picture, "list": [picture]})
    assert slimmed["text"] == "a" * 10
    assert slimmed["data"].endswith("bytes of inline data>") and slimmed["list"][0] == \
        slimmed["data"]


# --- the launch check ------------------------------------------------------------------

PROMPT = "Register the brain in job folder X.\n\nThe job.\nSecond line of the job.\n"


def _hook_run(job: Path, first: dict[str, Any], tmp_path: Path, monkeypatch, capsys,
              harness: str = "claude-code") -> dict[str, Any]:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_lines(first, {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash",
         "input": {"command": f"langslice-job {job} status"}}]}}))
    hook = {"conversationId": "c1", "transcriptPath": str(transcript)}
    assert _run_hook(hook, harness, monkeypatch, capsys) == "{}"
    return json.loads((job / "trace" / f"{harness}_c1.launch.json").read_text())


def _saved_launch(job: Path, text: str = PROMPT) -> None:
    folder = job / "logs" / "launches"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "000001_register.md").write_text(text)


def test_launch_check_verbatim_changed_and_unchecked(capsys, job, tmp_path, monkeypatch):
    user = {"type": "user", "message": {"role": "user", "content": PROMPT.rstrip() + "\n\n"}}
    # No saved launch file: not checked.
    checked = _hook_run(job, user, tmp_path, monkeypatch, capsys)
    assert checked["checked"] is False and "launch" in checked["reason"]
    _saved_launch(job)
    checked = _hook_run(job, user, tmp_path, monkeypatch, capsys)
    assert checked == {"checked": True, "verbatim": True, "launch": "000001_register.md",
                       "similarity": 1.0, "first_difference": None,
                       "expected_chars": len(PROMPT.rstrip()),
                       "got_chars": len(PROMPT.rstrip())}
    # Text blocks join; a summarised prompt is not verbatim and says where it differs.
    short = {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": "Register the brain in job folder X.\n\nThe job."}]}}
    checked = _hook_run(job, short, tmp_path, monkeypatch, capsys)
    assert checked["verbatim"] is False and checked["first_difference"] == len(
        "Register the brain in job folder X.\n\nThe job.")
    assert 0 < checked["similarity"] < 1
    # No first user message: not checked.
    none = {"type": "assistant", "message": {"role": "assistant", "content": "hi"}}
    assert _hook_run(job, none, tmp_path, monkeypatch, capsys)["checked"] is False


def test_launch_check_reads_an_antigravity_user_request(capsys, job, tmp_path, monkeypatch):
    _saved_launch(job)
    step = {"step_index": 0, "type": "USER_INPUT", "content": (
        f"<USER_REQUEST>\n{PROMPT.rstrip()}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nx\n"
        "</ADDITIONAL_METADATA>")}
    checked = _hook_run(job, step, tmp_path, monkeypatch, capsys, harness="antigravity")
    assert checked["checked"] is True and checked["verbatim"] is True


@pytest.mark.parametrize("tail", ['. Call start_job(image_folder="{job}") first: it opens',
                                  " with the langslice-job command line."])
def test_an_agent_that_never_called_lands_in_the_job_its_launch_names(
        capsys, job, tmp_path, monkeypatch, tail):
    text = f"Register the brain in job folder {job}{tail.format(job=job)}\n\nThe job."
    _saved_launch(job, text)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_lines({"type": "user", "message": {"role": "user", "content": text}}))
    hook = {"conversationId": "c2", "transcriptPath": str(transcript)}
    assert _run_hook(hook, "claude-code", monkeypatch, capsys) == "{}"
    checked = json.loads((job / "trace" / "claude-code_c2.launch.json").read_text())
    assert checked["verbatim"] is True
