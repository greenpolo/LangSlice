"""The local activity surface must preserve content and remain read-only."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from langslice.hosts.integrations import abba_chat as chat


def test_history_preserves_summary_deltas_and_ignores_private_fields():
    history = chat.ChatHistory()
    history.add({"kind": "session", "model": "model-2", "text": "private instruction"})
    history.add({"kind": "reasoning", "text": "Slices **1–4**", "thought_signature": "secret"})
    history.add({"kind": "reasoning", "text": " look anterior."})
    history.add(
        {
            "kind": "tool_start",
            "name": "look",
            "execution_id": "one",
            "args": {"ids": [1, 2], "encrypted_content": "secret"},
        }
    )
    history.add(
        {
            "kind": "tool_end",
            "name": "look",
            "execution_id": "one",
            "response": {"status": "ok", "nested": {"thought_signature": "secret"}},
        }
    )
    result = history.snapshot()
    assert [e["text"] for e in result["events"][:2]] == ["Slices **1–4**", " look anterior."]
    assert "secret" not in json.dumps(result)
    assert "instruction" not in json.dumps(result)
    assert result["events"][2]["execution_id"] == result["events"][3]["execution_id"]


def test_history_bounds_events_and_reports_replay_gap(monkeypatch):
    monkeypatch.setattr(chat, "MAX_EVENTS", 2)
    history = chat.ChatHistory()
    for i in range(5):
        history.add({"kind": "text", "text": str(i)})
    result = history.snapshot(1)
    assert result["reset"]
    assert result["trimmed"]
    assert [e["text"] for e in history.snapshot()["events"]] == ["3", "4"]
    assert history.snapshot(4)["events"][0]["text"] == "4"


def test_images_are_exact_bounded_bytes_not_embedded_in_event_json(monkeypatch):
    monkeypatch.setattr(chat, "MAX_IMAGE_BYTES", 8)
    history = chat.ChatHistory()
    history.add(
        {
            "kind": "seed",
            "text": "instruction",
            "images": [
                {"data": b"12345", "mime_type": "image/png", "label": "Slice 1"},
                {"data": b"67890", "mime_type": "image/png", "label": "Slice 2"},
                {"data": b"<svg>", "mime_type": "image/svg+xml", "label": "invalid"},
            ],
        }
    )
    assert len(history.images) == 1
    assert next(iter(history.images.values())) == (b"67890", "image/png")
    result = history.snapshot()
    assert len(result["events"][0]["images"]) == 2
    assert "12345" not in json.dumps(result)
    assert "instruction" not in json.dumps(result)
    assert "invalid" not in json.dumps(result)


def test_server_requires_token_host_and_exposes_no_mutations():
    history = chat.ChatHistory()
    history.add({"kind": "text", "text": "Working"})
    server = chat.ChatServer(history)
    try:
        with urlopen(server.url + "events", timeout=3) as response:
            assert json.load(response)["events"][0]["text"] == "Working"
            assert response.headers["Cache-Control"] == "no-store"
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with urlopen(server.url, timeout=3) as response:
            assert b"chat.js" in response.read()
        for request in [
            server.url.replace(server.token, "wrong") + "events",
            Request(server.url + "events", headers={"Host": "evil.example"}),
            server.url + "../abba_chat.py",
            Request(server.url + "events", data=b"{}", method="POST"),
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=3)
            assert error.value.code in {404, 501}
    finally:
        server.close()


def test_browser_missing_falls_back_to_existing_viewer(monkeypatch):
    from langslice.hosts.integrations import abba_activity

    monkeypatch.setattr(chat, "_browser", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    marker = object()
    monkeypatch.setattr(abba_activity, "ActivityWindow", lambda **kwargs: marker)
    assert chat.create_activity_window() is marker


def test_windows_browser_searches_standard_install_locations(monkeypatch, tmp_path):
    browser = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    browser.parent.mkdir(parents=True)
    browser.touch()
    monkeypatch.setattr(chat.shutil, "which", lambda _: None)
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert chat._browser(windows=True) == str(browser)


def test_disposed_window_ignores_future_events_and_show():
    window = chat.ChatWindow.__new__(chat.ChatWindow)
    window.closed = True
    window.on_event({"kind": "text", "text": "ignored"})
    window.show()
    window.dispose()


def test_browser_launch_uses_private_profile_and_abba_display(monkeypatch, tmp_path):
    class FakeServer:
        url = "http://127.0.0.1:1234/token/"
        closed = False

        def __init__(self, history):
            pass

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.returncode is None:
                raise chat.subprocess.TimeoutExpired(self.args, timeout)
            return self.returncode

        def terminate(self):
            self.returncode = 0

    monkeypatch.setattr(chat, "ChatServer", FakeServer)
    monkeypatch.setattr(chat, "_browser", lambda: "/usr/bin/google-chrome")
    monkeypatch.setattr(chat, "_arrange", lambda parent: (1200, 0, 600, 1000))
    monkeypatch.setattr(chat.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "private"))
    monkeypatch.setattr(chat.subprocess, "Popen", Process)
    monkeypatch.setenv("DISPLAY", ":108")
    window = chat.ChatWindow()
    args = window.process.args
    assert ("--ozone-platform=x11" in args) == (chat.os.name == "posix")
    assert "--window-position=1200,0" in args
    assert f"--user-data-dir={tmp_path / 'private'}" in args
    assert not any("no-sandbox" in arg for arg in args)
    window.on_event({"kind": "text", "text": "Still running"})
    first = window.process
    window.show()  # already open: no second window
    assert window.process is first
    window.process.returncode = 0  # Closing the window does not close the server.
    window.show()
    assert window.process is not first
    assert not window.server.closed
    assert window.history.snapshot()["events"][0]["text"] == "Still running"
    window.dispose()
    assert window.server.closed
    assert window.process.returncode == 0
