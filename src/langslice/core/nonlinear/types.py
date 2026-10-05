"""The image model's request and reply: what a door's image call takes and returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class SegmentationGenerationRequest:
    """One image-model edit: the section to edit, its references and the prompt."""

    #: The references, in prompt order: they follow the slice image as
    #: Image 2..N. The prompt describes what each one is; providers just
    #: deliver them in this order.
    reference_images: list[Image.Image]
    slice_image: Image.Image
    prompt: str
    provider: str = "gemini-api"
    model: str | None = None
    #: Transport fields the provider adapter reads (``providers/images.py``);
    #: no core caller sets them.
    route: str | None = None
    review_model: str | None = None
    #: Registration is always an EDIT of the slice image (pixel-aligned
    #: output); each transport translates this its own way.
    mode: str = "edit"
    openai_image_route: str = "images"
    #: The output tier the transport passes on (Gemini's image size, the
    #: OpenAI images quality); no core caller sets it.
    thinking_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedSegmentation:
    """The image model's reply to one :class:`SegmentationGenerationRequest`."""

    image: Image.Image
    provider: str
    model: str
    route: str
    revised_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
