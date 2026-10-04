"""`langslice abba`'s Python listener: the connector's run messages feed the
passive viewer and log (fakes; no JVM in the plain environment)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from langslice.hosts.integrations import abba_launch
from langslice.hosts.integrations.abba_launch import RunListener, connector_jar, read_views


class _Follower:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self.mapping = dict(mapping)
        self.events: list[dict[str, Any]] = []
        self.finished = 0

    def on_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def finish(self) -> None:
        self.finished += 1


class _Log:
    closed = False

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def on_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _listener(slices: dict[str, Any]) -> tuple[RunListener, list[_Follower], list[_Log]]:
    followers: list[_Follower] = []
    logs: list[_Log] = []

    def follower(mapping: Any) -> _Follower:
        followers.append(_Follower(mapping))
        return followers[-1]

    def log() -> _Log:
        logs.append(_Log())
        return logs[-1]

    listener = RunListener(slices=lambda: slices, follower_factory=follower, log_factory=log,
                           view_wait_s=0.0)
    return listener, followers, logs


def _agent(event: dict[str, Any]) -> str:
    return json.dumps({"kind": "agent_event", "event": event})


def test_agent_events_reach_the_viewer_and_the_log_with_pictures_from_the_views(tmp_path):
    picture = tmp_path / "views" / "000003_view_slices" / "view.jpg"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"\xff\xd8jpeg")
    slices = {"section_0001.tif": "slice-a", "section_0002.tif": "slice-b"}
    listener, followers, logs = _listener(slices)
    start = {"kind": "tool_start", "name": "view_slices", "execution_id": "x",
             "target_ids": ["section_0002.tif"]}
    end = {**start, "kind": "tool_end", "views": [str(picture)]}
    listener.handle(_agent(start))
    listener.handle(_agent(end))
    [follower] = followers
    assert follower.mapping == slices and follower.events == [start, end]
    [log] = logs
    assert log.events[0] == start
    [image] = log.events[1]["images"]
    assert image == {"data": b"\xff\xd8jpeg", "mime_type": "image/jpeg",
                     "label": "000003_view_slices"}
    assert "images" not in end  # the follower's copy is untouched


def test_a_new_section_map_is_a_new_run_and_the_end_finishes_the_follower():
    slices: dict[str, Any] = {"a.tif": 1}
    listener, followers, _logs = _listener(slices)
    listener.handle(_agent({"kind": "seed", "views": []}))
    slices.clear()
    slices.update({"b.tif": 2})
    listener.handle(_agent({"kind": "seed", "views": []}))
    assert [f.mapping for f in followers] == [{"a.tif": 1}, {"b.tif": 2}]
    assert followers[0].finished == 1
    listener.handle(json.dumps({"kind": "run_finished"}))
    assert followers[1].finished == 1 and listener.follower is None


def test_failing_viewers_never_reach_the_run_and_log_messages_are_progress():
    def broken(_mapping: Any) -> Any:
        raise RuntimeError("no display")

    logs: list[_Log] = []
    listener = RunListener(slices=lambda: {"a.tif": 1}, follower_factory=broken,
                           log_factory=lambda: logs.append(_Log()) or logs[-1], view_wait_s=0.0)
    listener.handle(_agent({"kind": "tool_start", "name": "status", "target_ids": []}))
    listener.handle(json.dumps({"kind": "log", "message": "fitting s0"}))
    assert logs[0].events[-1] == {"kind": "progress", "text": "fitting s0"}


def test_the_java_thread_only_queues_messages():
    """accept() returns at once; one background thread handles messages in order."""
    listener, followers, _logs = _listener({"a.tif": 1})
    for index in range(5):
        listener.accept(_agent({"kind": "tool_start", "name": "view_slices",
                                "execution_id": str(index), "target_ids": ["a.tif"]}))
    listener.accept("not json")  # a bad message is logged and skipped
    listener.close()
    assert [event["execution_id"] for event in followers[0].events] == list("01234")


def test_missing_or_unlisted_views_are_skipped(tmp_path):
    assert read_views([str(tmp_path / "missing.jpg"), 3], wait_s=0.0) == []
    assert read_views(None) == []


def test_the_connector_jar_comes_from_the_environment_or_the_build_output(tmp_path, monkeypatch):
    monkeypatch.delenv(abba_launch.CONNECTOR_JAR_ENV, raising=False)
    monkeypatch.setattr(abba_launch, "repository_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="LANGSLICE_CONNECTOR_JAR"):
        connector_jar()
    target = tmp_path / "connectors" / "fiji" / "target"
    target.mkdir(parents=True)
    (target / "langslice-fiji-0.1-sources.jar").write_bytes(b"PK")
    built = target / "langslice-fiji-0.1.jar"
    built.write_bytes(b"PK")
    assert connector_jar() == built.resolve()
    elsewhere = tmp_path / "other.jar"
    elsewhere.write_bytes(b"PK")
    monkeypatch.setenv(abba_launch.CONNECTOR_JAR_ENV, str(elsewhere))
    assert connector_jar() == elsewhere.resolve()
    monkeypatch.setenv(abba_launch.CONNECTOR_JAR_ENV, str(tmp_path / "gone.jar"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        connector_jar()


def test_abba_0_24_dependencies_are_pinned():
    assert "ch.epfl.biop:ImageToAtlasRegister:0.24.1" in abba_launch.ABBA_JAVA_DEPENDENCIES
    assert Path(abba_launch.__file__).resolve().parents[4] == abba_launch.repository_root()


def test_the_viewer_opens_only_when_the_run_asked_for_it_and_failures_are_logged():
    listener, followers, logs = _listener({"a.tif": 1})
    listener.handle(json.dumps({"kind": "run_started", "mode": "claude", "viewer": False,
                                "sections": {"a.tif": "slice 1"}}))
    listener.handle(_agent({"kind": "tool_start", "name": "view_slices", "target_ids": ["a.tif"]}))
    assert followers == []
    listener.handle(json.dumps({"kind": "applied", "applied": [], "angles": False,
                                "failed": {"a.tif": "a newer registration"}}))
    assert "a.tif: a newer registration" in logs[0].events[-1]["text"]
    listener.handle(json.dumps({"kind": "run_started", "viewer": True}))
    assert len(followers) == 1
    listener.handle(json.dumps({"kind": "run_finished", "message": "Done"}))
    assert logs[0].events[-1] == {"kind": "status", "text": "Done"}
    assert followers[0].finished == 1
