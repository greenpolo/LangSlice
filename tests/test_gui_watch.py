"""The GUI's job watcher: file tailing, whole-file rereads, the sections table,
undo labels and the job list (``langslice.hosts.gui.watch``)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from langslice.core.state import SliceState, StackState
from langslice.hosts.gui import watch
from langslice.hosts.gui.watch import JobList, JobWatcher
from langslice.job.checkpoint import write_checkpoint, write_json_atomic


def _append(path: Path, *records: dict | str, end: str = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write((record if isinstance(record, str) else json.dumps(record)) + end)


def _job(tmp_path: Path) -> Path:
    folder = tmp_path / "images" / "langslice"
    folder.mkdir(parents=True)
    write_json_atomic(str(folder / "job.json"), {
        "format_version": 1, "image_folder": "..", "notes": "Nissl, every 4th section",
        "spec": {"atlas": "allen_mouse_25um", "plane": "coronal", "tasks": ["position"],
                 "model": "openai-oauth/gpt-5"}})
    return folder


def _state(folder: Path, positions: list[float | None], *, submitted: bool = False) -> None:
    slices = [SliceState(id=f"s{i:03d}.png", index_original=i, index_corrected=i,
                         position_mm=p) for i, p in enumerate(positions)]
    write_checkpoint(StackState(atlas="allen_mouse_25um", slices=slices,
                                submitted=submitted), str(folder / "state.json"))


def _view(folder: Path, seq: int, tool: str, sections: list[str], call: int = 1) -> dict:
    path = f"views/{seq:06d}_{tool}"
    (folder / path).mkdir(parents=True, exist_ok=True)
    (folder / path / "view.jpg").write_bytes(b"\xff\xd8jpeg" + bytes([seq]))
    (folder / path / "view.json").write_text(json.dumps({"caption": f"{tool} picture {seq}"}))
    return {"seq": seq, "name": f"{seq:06d}_{tool}", "path": path, "tool": tool, "call": call,
            "sections": sections, "mode": None, "layers": False}


def _snap(watcher: JobWatcher, since: int = 0) -> dict:
    watcher.poll(force=True)
    return watcher.snapshot(since=since, generation=watcher.generation)


def test_appended_lines_arrive_once_in_order(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    events = folder / "logs" / "events.jsonl"
    _append(events, {"kind": "open", "resumed": False, "at": "2026-10-28T10:00:00+00:00"})
    watcher = JobWatcher(folder)
    first = _snap(watcher)
    assert [i["text"] for i in first["items"]] == ["Job opened"]
    _append(events, {"kind": "tool_start", "name": "position_sections", "execution_id": "e1",
                     "door": "cli", "target_ids": ["s000.png"], "args": {"sections": []},
                     "at": "2026-10-28T10:00:01+00:00"})
    second = _snap(watcher, first["cursor"])
    assert [i["kind"] for i in second["items"]] == ["tool"]
    assert second["status"]["word"] == "Running"
    assert second["status"]["sentence"].startswith("Position sections on s000.png")
    assert second["header"]["door"] == "cli"
    assert _snap(watcher, second["cursor"])["items"] == []


def test_a_half_written_line_waits_for_its_end(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    views = folder / "views.jsonl"
    record = _view(folder, 1, "look", ["s000.png"])
    text = json.dumps(record)
    _append(views, text[:20], end="")
    watcher = JobWatcher(folder)
    assert [i for i in _snap(watcher)["items"] if i["kind"] == "picture"] == []
    _append(views, text[20:])
    pictures = [i for i in _snap(watcher)["items"] if i["kind"] == "picture"]
    assert [(p["number"], p["caption"]) for p in pictures] == [(1, "look picture 1")]
    assert watcher.picture(1) == folder / "views" / "000001_look" / "view.jpg"
    assert watcher.picture(2) is None


def test_state_replaced_whole_is_reread(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    _state(folder, [None, None])
    watcher = JobWatcher(folder)
    snap = _snap(watcher)
    assert [r["position_mm"] for r in snap["sections"]] == [None, None]
    version = snap["state_version"]
    _state(folder, [2.1, 2.4], submitted=True)  # a temp file renamed over the old one
    watcher.poll(force=True)
    snap = watcher.snapshot(since=snap["cursor"], generation=watcher.generation,
                            state_version=version)
    assert [r["position_mm"] for r in snap["sections"]] == [2.1, 2.4]
    assert snap["status"]["word"] == "Submitted"
    # Unchanged since: the tables are not sent again.
    again = watcher.snapshot(since=snap["cursor"], generation=watcher.generation,
                             state_version=snap["state_version"])
    assert "sections" not in again


def test_a_trace_appearing_mid_run_is_read(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    watcher = JobWatcher(folder)
    assert _snap(watcher)["items"] == []
    trace = folder / "trace" / "linear_stack_1234abcd.jsonl"
    _append(trace,
            {"kind": "session", "model": "claude-opus-5-5", "at": "2026-10-28T10:00:00"},
            {"kind": "model", "thought": "Matching s004 to 2.1 mm.", "text": "Placing s004.",
             "function_calls": [], "at": "2026-10-28T10:00:05"},
            {"kind": "tool_result", "name": "look", "response": {}, "at": "2026-10-28T10:00:06"})
    items = _snap(watcher)["items"]
    assert [(i["kind"], i["text"]) for i in items] == [
        ("note", "Agent session started (claude-opus-5-5)"),
        ("reasoning", "Matching s004 to 2.1 mm."), ("agent", "Placing s004.")]
    assert watcher.header()["model"] == "claude-opus-5-5"
    _append(trace, {"kind": "summary", "submitted": False, "spend": {"model": "m", "tokens": {
        "output": 5}, "usd_estimate": 0.5}, "at": "2026-10-28T10:01:00"})
    snap = _snap(watcher)
    assert snap["status"]["word"] == "Failed"
    assert snap["spend"]["usd_estimate"] == 0.5


def test_trace_and_events_read_together_are_in_time_order(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    _append(folder / "logs" / "events.jsonl",
            {"kind": "tool_start", "name": "look", "execution_id": "a", "door": "agent",
             "target_ids": [], "args": {"mode": "atlas"}, "at": "2026-10-28T10:00:05+00:00"},
            {"kind": "tool_end", "name": "look", "execution_id": "a", "door": "agent",
             "target_ids": [], "args": {}, "step": 0, "views": [], "failed": False,
             "status": "ok", "at": "2026-10-28T10:00:06+00:00"})
    utc = "2026-10-28T10:00:05+00:00"
    from datetime import datetime
    local = datetime.fromisoformat(utc).astimezone().replace(tzinfo=None).isoformat()
    _append(folder / "trace" / "t.jsonl",
            {"kind": "model", "thought": "Look at the atlas.", "at": local})
    kinds = [(i["kind"], i.get("phase")) for i in _snap(JobWatcher(folder))["items"]]
    assert kinds == [("reasoning", None), ("tool", "start"), ("tool", "end")]


def test_missing_files_and_folder(tmp_path: Path) -> None:
    watcher = JobWatcher(tmp_path / "not_yet" / "langslice")
    snap = _snap(watcher)
    assert snap["items"] == [] and snap["sections"] == []
    assert snap["status"] == {"word": "Waiting",
                              "sentence": "Waiting for the job folder to appear."}
    assert snap["spend"] is None and snap["notes"] == {"user": "", "run": []}
    assert snap["header"]["exists"] is False


def test_a_state_this_version_does_not_read_shows_raw_fields(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    write_json_atomic(str(folder / "state.json"), {
        "format_version": 3, "appearance": {}, "cutting_angles_deg": None,
        "slices": [
            {"id": "b.png", "index_corrected": 1, "position_mm": 4.5, "flip": True,
             "transform": {"kind": "elastix"}, "deformation": {"steps": [{"trace": "x"}]},
             "damaged_regions": ["CB"], "cutting_angles_deg": {"pitch": 2.0, "yaw": -0.5}},
            {"id": "a.png", "index_corrected": 0, "position_mm": 4.3, "transform": None,
             "deformation": None}]})
    snap = _snap(JobWatcher(folder))
    assert "raw fields" in snap["sections_note"]
    assert [(r["order"], r["id"]) for r in snap["sections"]] == [(1, "a.png"), (2, "b.png")]
    b = snap["sections"][1]
    assert (b["flip"], b["transform"], b["deformation"], b["damaged"]) == (
        True, "elastix", "trace_borders", True)
    assert b["angles"] == {"pitch": 2.0, "yaw": -0.5}


def _end(name: str, step: int, sections: tuple[str, ...] = (), failed: bool = False) -> dict:
    return {"kind": "tool_end", "name": name, "execution_id": f"{name}{step}", "door": "agent",
            "target_ids": list(sections), "args": "{\"sections\": [", "args_truncated": True,
            "step": step, "views": [], "failed": failed, "status": "ok",
            "at": "2026-10-28T10:00:00+00:00"}


def test_undo_steps_are_labelled_by_the_call_that_made_each_depth(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    events = folder / "logs" / "events.jsonl"
    _append(events, _end("position_sections", 1, ("s000.png",)), _end("look", 1),
            _end("interactive_transform", 2, ("s001.png",)), _end("note", 3),
            _end("undo", 2), _end("elastix_affine", 2, ("s001.png",), failed=True))
    write_json_atomic(str(folder / "history" / "index.json"), {
        "format_version": 1, "undo": ["step-000001.json", "step-000002.json"],
        "redo": ["step-000003.json"]})
    watcher = JobWatcher(folder)
    snap = _snap(watcher)
    assert snap["history"] == {
        "undo": ["Position sections (s000.png)", "Interactive transform (s001.png)"],
        "redo": ["Note"]}
    tool = next(i for i in snap["items"] if i["kind"] == "tool")
    assert tool["args"] == "{\"sections\": ["  # a truncated argument string passes through
    _append(events, _end("redo", 3), _end("mark_damage", 4, ("s002.png",)))
    write_json_atomic(str(folder / "history" / "index.json"), {
        "format_version": 1, "redo": [],
        "undo": [f"step-00000{n}.json" for n in (1, 2, 3, 4)]})
    assert _snap(watcher)["history"]["undo"][2:] == ["Note", "Mark damage (s002.png)"]


def test_at_the_history_cap_a_write_drops_the_oldest_label(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    depth = watch.UNDO_DEPTH
    _append(folder / "logs" / "events.jsonl",
            *[_end("interactive_transform", d, (f"s{d:03d}.png",)) for d in range(1, depth + 1)],
            _end("look", depth), _end("note", depth))
    write_json_atomic(str(folder / "history" / "index.json"), {
        "format_version": 1, "redo": [],
        "undo": [f"step-{n:06d}.json" for n in range(2, depth + 2)]})
    undo = _snap(JobWatcher(folder))["history"]["undo"]
    assert len(undo) == depth
    assert undo[0] == "Interactive transform (s002.png)" and undo[-1] == "Note"


def test_pictures_named_by_a_path_outside_the_folder_are_refused(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "view.jpg").write_bytes(b"x")
    _append(folder / "views.jsonl", {"seq": 1, "path": "../../secret", "tool": "look"})
    watcher = JobWatcher(folder)
    _snap(watcher)
    assert watcher.picture(1) is None


def test_the_job_list_reads_given_folders_and_recent_jobs(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
    given = _job(tmp_path)
    other = tmp_path / "other" / "langslice"
    shutil.copytree(given, other)
    recent = tmp_path / "recent_jobs.jsonl"
    _append(recent, {"folder": str(other), "image_folder": str(other.parent),
                     "atlas": "allen_mouse_25um", "at": "2026-10-28T10:00:00+00:00"},
            {"folder": str(tmp_path / "gone")}, "not json", {"folder": str(given)})
    jobs = JobList([given], recent=recent)
    listing = jobs.listing()
    assert [(j["index"], Path(j["folder"]), j["given"]) for j in listing] == [
        (0, given, True), (1, other, False)]
    assert listing[1]["name"] == "other"
    assert jobs.get(1) is not None and jobs.get(2) is None
    monkeypatch.setenv(watch.RECENT_ENV, "off")
    assert watch.recent_jobs_file() is None
    monkeypatch.setenv(watch.RECENT_ENV, str(recent))
    assert watch.recent_jobs_file() == recent
    monkeypatch.delenv(watch.RECENT_ENV)
    assert watch.recent_jobs_file() == Path.home() / ".langslice" / "recent_jobs.jsonl"


def test_a_truncated_file_starts_the_watch_over(tmp_path: Path) -> None:
    folder = _job(tmp_path)
    events = folder / "logs" / "events.jsonl"
    _append(events, {"kind": "open", "at": "2026-10-28T10:00:00+00:00"},
            {"kind": "open", "resumed": True, "at": "2026-10-28T10:00:01+00:00"})
    watcher = JobWatcher(folder)
    first = _snap(watcher)
    events.write_text(json.dumps({"kind": "open", "at": "2026-10-28T11:00:00+00:00"}) + "\n")
    watcher.poll(force=True)
    snap = watcher.snapshot(since=first["cursor"], generation=first["generation"])
    assert snap["generation"] != first["generation"]
    assert [i["text"] for i in snap["items"]] == ["Job opened"]


@pytest.mark.slow
def test_a_real_finished_job_folder(tmp_path: Path) -> None:
    source = Path(os.environ.get(
        "LANGSLICE_GUI_REPLAY_JOB",
        "/home/walshlab/cc_sandbox/harness_cc_vs_ls_20261027/NM01_GLT1a/images/langslice"))
    if not (source / "views.jsonl").is_file():
        pytest.skip("no real job folder here (LANGSLICE_GUI_REPLAY_JOB)")
    folder = tmp_path / "langslice"
    shutil.copytree(source, folder)
    lines = (folder / "views.jsonl").read_text().splitlines()
    (folder / "views.jsonl").write_text("")
    watcher = JobWatcher(folder)
    _snap(watcher)
    for line in lines:
        _append(folder / "views.jsonl", line)
    snap = _snap(watcher)
    assert len([i for i in snap["items"] if i["kind"] == "picture"]) == len(lines)
    assert snap["sections"]
