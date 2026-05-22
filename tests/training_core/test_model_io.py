from __future__ import annotations

from langslice_training.model_io import RenderedMessage, RenderedMessageBundle, render_chat_template


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        return_tensors: str | None = None,
    ) -> str:
        self.calls.append((messages, return_tensors))
        return f"rendered:{len(messages)}"


def test_render_chat_template_delegates_to_processor() -> None:
    bundle = RenderedMessageBundle(
        messages=[
            RenderedMessage(
                role="user",
                content=[{"type": "text", "text": "Estimate AP position."}],
            )
        ],
        image_paths=["images/query.png"],
        tools=[],
        metadata={"plane": "coronal"},
    )
    processor = _FakeProcessor()

    rendered = render_chat_template(processor, bundle)

    assert rendered == "rendered:1"
    assert len(processor.calls) == 1
    called_messages, called_return_tensors = processor.calls[0]
    assert called_messages[0]["role"] == "user"
    assert called_return_tensors is None

