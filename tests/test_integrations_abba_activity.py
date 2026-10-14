"""Companion history and display contracts, without a JVM or model calls."""

from types import SimpleNamespace

from langslice.agent.live import LiveEvents
from langslice.hosts.integrations import abba_activity as activity


def _image(data: bytes, label: str = "Section") -> dict:
    return {"data": data, "mime_type": "image/png", "label": label}


def test_history_rejects_invalid_images_and_copies_only_display_fields(monkeypatch):
    monkeypatch.setattr(activity, "MAX_IMAGE_BYTES", 10)
    history = activity.ActivityHistory()
    image = {**_image(b"abc"), "encrypted_content": "private", "extra": object()}
    history.add(
        "First",
        [
            {},
            _image(b""),
            {"data": "not bytes"},
            _image(b"x" * 11),
            image,
        ],
    )
    image["label"] = "Changed after delivery"
    assert history.byte_count == 3
    assert len(history.batches) == 1
    assert history.batches[0].images == [_image(b"abc")]
    history.add("Empty", [_image(b"")])
    assert len(history.batches) == 1


def test_history_evicts_oldest_batch_under_byte_and_count_limits(monkeypatch):
    monkeypatch.setattr(activity, "MAX_IMAGE_BYTES", 8)
    monkeypatch.setattr(activity, "MAX_IMAGE_BATCHES", 2)
    history = activity.ActivityHistory()
    history.add("First", [_image(b"aaa")])
    history.add("Second", [_image(b"bbb")])
    history.add("Third", [_image(b"ccc")])
    assert [batch.label for batch in history.batches] == ["Second", "Third"]
    assert history.byte_count == 6
    history.add("Fourth", [_image(b"d" * 7)])
    assert [batch.label for batch in history.batches] == ["Fourth"]
    assert history.byte_count == 7
    assert history.removed >= 3


def test_huge_batch_keeps_newest_images_within_total_budget(monkeypatch):
    # Thousands of images exercise a large batch without allocating large assets.
    monkeypatch.setattr(activity, "MAX_IMAGE_BYTES", 23)
    history = activity.ActivityHistory()
    history.add("Huge", [_image(b"four", str(index)) for index in range(3000)])
    assert len(history.batches) == 1
    assert [image["label"] for image in history.batches[0].images] == [
        "2995",
        "2996",
        "2997",
        "2998",
        "2999",
    ]
    assert history.byte_count == 20
    assert history.byte_count == sum(
        len(image["data"]) for batch in history.batches for image in batch.images
    )
    assert history.removed > 0


def test_live_events_never_show_private_fields_or_markup():
    class PrivateObject:
        def __repr__(self):
            raise AssertionError("A provider object's repr must never be requested")

    emitted = []
    live = LiveEvents(emitted.append)
    protected = {
        "safe": "visible result",
        "encrypted_content": "secret ciphertext",
        "nested": {"thought_signature": "secret signature", "safe": "visible nested"},
        "provider": PrivateObject(),
    }
    live.event(
        SimpleNamespace(
            partial=False,
            content=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        text="Public summary", thought=True, thought_signature=b"private bytes"
                    ),
                    SimpleNamespace(
                        function_call=SimpleNamespace(name="inspect", args=protected, id="c1")
                    ),
                    SimpleNamespace(
                        function_response=SimpleNamespace(
                            name="inspect",
                            response=protected,
                            id="c1",
                        )
                    ),
                ]
            ),
        )
    )
    display = "".join(activity.format_event(event)[0] for event in emitted)
    assert "Public summary" in display
    assert "secret" not in display
    assert "private bytes" not in display
    assert "encrypted_content" not in display
    assert "thought_signature" not in display
    # Extra provider fields and unknown event kinds are never stringified.
    assert activity.format_event({"kind": "provider", "payload": PrivateObject()})[0] == ""
    public = {"kind": "text", "text": "Public", "provider": PrivateObject()}
    assert activity.format_event(public)[0] == "Public"
    # The transcript renders public markdown and never interprets markup.
    rendered = activity.transcript_html(
        [("**Comparing slices**\n<script>hidden</script>", "reasoning")])
    assert "<b>Comparing slices</b>" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
