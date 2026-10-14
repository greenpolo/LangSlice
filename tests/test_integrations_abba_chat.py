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


