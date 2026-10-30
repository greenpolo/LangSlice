"""The image model's request and reply, and the image model as the core takes it.

A door resolves the image model (a provider's transport, or a script's own
function) and hands the core an :class:`ImageModel`; its ``call`` is an
:class:`ImageCall`, the one edit. The core and the tools name these types,
never a provider's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass
class BorderTraceRequest:
    """Hold one image-model edit: the section, its references, the prompt."""

    #: The references, in prompt order: they follow the slice image as
    #: Image 2..N. The prompt describes what each one is; providers just
    #: deliver them in this order.
    reference_images: list[Image.Image]
    slice_image: Image.Image
    prompt: str
    provider: str = "gemini-api"
    model: str | None = None
    #: Registration is always an EDIT of the slice image (pixel-aligned
    #: output); each transport translates this its own way.
    mode: str = "edit"
    #: The output tier the transport passes on (Gemini's image size, the
    #: OpenAI images quality).
    size_tier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedSegmentation:
    """The image model's reply to one :class:`BorderTraceRequest`."""

    image: Image.Image
    provider: str
    model: str
    route: str
    revised_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageCall(Protocol):
    """Make one image edit: the request in, one image out."""

    def __call__(
        self, request: BorderTraceRequest, /
    ) -> GeneratedSegmentation:
        """Edit the section image as the request says.

        Parameters
        ----------
        request
            The prompt, the images in order, the provider and the model.

        Returns
        -------
        GeneratedSegmentation
            The edited image and what made it.
        """
        ...


@dataclass(frozen=True)
class ImageModel:
    """An image model as the operations receive it: resolved, never chosen there.

    ``provider`` is the canonical access method (it selects the prompt's
    GPT or Gemini wording and keys saved replies; ``"custom"`` for a model
    of the caller's own), ``model`` the image model (None: the provider's
    own default), ``call`` the one edit. A door (the toolbox binding, the
    engine, MCP, the CLI, a host plugin) builds this from a provider's name;
    a test or a script passes its own ``call``.

    The rest makes it a model PROFILE (a prompt written for one model):
    ``prompt`` the base prompt written for this model (None: LangSlice's own
    for ``provider``,
    :func:`langslice.core.fits.trace.prompts.border_correction_tool_prompt`;
    ``{plane}`` in it becomes the section plane), ``photograph_first`` the
    attachment order that prompt describes (True: the clean photograph is Image
    1 and the placed borders Image 2; False: the other way round; None: the
    order of ``provider``'s own prompt), ``profile`` its name, and ``tested``
    False for a prompt or model the project has not measured: every trace made
    with it is marked ``untested``.
    """

    provider: str
    model: str | None
    call: ImageCall
    prompt: str | None = None
    photograph_first: bool | None = None
    profile: str | None = None
    tested: bool = True
