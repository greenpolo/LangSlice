"""Model-neutral message/image bundle helpers for training pipelines."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class RenderedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: str
    content: list[dict[str, Any]] | str


class RenderedMessageBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    messages: list[RenderedMessage] = Field(min_length=1)
    image_paths: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatTemplateProcessor(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = ...,
        tokenize: bool = ...,
        return_tensors: str | None = ...,
    ) -> Any: ...


def render_chat_template(
    processor: ChatTemplateProcessor,
    bundle: RenderedMessageBundle,
    *,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    return_tensors: str | None = None,
) -> Any:
    rendered_messages = [message.model_dump(mode="python") for message in bundle.messages]
    return processor.apply_chat_template(
        rendered_messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
        return_tensors=return_tensors,
    )

