"""Model profiles: an image model and the prompt written for it.

Different image models need different prompts, so the trace's model and its
prompt travel together as one :class:`~langslice.providers.registry.ImageModel`
(its ``prompt``, ``photograph_first``, ``profile`` and ``tested`` fields).

- A built-in profile is a provider's own model with LangSlice's prompt for it
  (:func:`langslice.core.nonlinear.prompts.border_correction_tool_prompt`,
  the GPT wording for the OpenAI providers, the general wording otherwise):
  ``image_model("openai-oauth")``.
- A custom profile names a model (a provider and model name, an
  ``ImageModel``, any object with a ``call``, or a plain function) and,
  optionally, the prompt written for it (text or a text file). It is marked
  untested: every trace it makes records ``profile`` and ``untested``.

:func:`default_prompt` gives LangSlice's own prompt for a provider, the text
to start a custom prompt from. Resolving a profile loads no model client
(the transport is imported when the model is called).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.provider_names import CUSTOM_PROVIDER, canonical_provider
from langslice.providers.registry import CANONICAL_PROVIDERS, ImageModel, resolve_image_model

if TYPE_CHECKING:
    from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest

#: The name a custom profile gets when the caller gives none.
CUSTOM_PROFILE = "custom"


def default_prompt(provider: str = "openai-oauth", plane: str = "coronal") -> str:
    """LangSlice's own trace prompt for *provider* on
    a *plane* section: what a built-in profile sends, and the text to start
    a custom prompt from. A custom prompt may write ``{plane}`` where the
    plane belongs."""
    from typing import cast

    from langslice.core.nonlinear.prompts import border_correction_tool_prompt
    from langslice.core.space import Plane

    return border_correction_tool_prompt(cast(Plane, plane), provider=canonical_provider(provider))


def _read_prompt(prompt: str | None, prompt_file: str | os.PathLike[str] | None) -> str | None:
    if prompt is not None and prompt_file is not None:
        raise ValueError("Give the prompt as text (prompt=) or as a file (prompt_file=), not both")
    if prompt_file is not None:
        prompt = Path(os.path.expanduser(os.fspath(prompt_file))).read_text(encoding="utf-8")
    if prompt is not None:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("The profile's prompt is empty")
    return prompt


def _wrapped_call(function: Callable[..., Any], provider: str, model: str | None,
                  ) -> Callable[[SegmentationGenerationRequest], GeneratedSegmentation]:
    """A caller's own edit function as an ``ImageCall``: it may answer with a
    ``GeneratedSegmentation`` or with the edited picture itself (a PIL image
    or an array)."""

    def call(request: SegmentationGenerationRequest) -> GeneratedSegmentation:
        from PIL import Image

        from langslice.core.nonlinear.types import GeneratedSegmentation

        reply = function(request)
        if isinstance(reply, GeneratedSegmentation):
            return reply
        if not isinstance(reply, Image.Image):
            try:
                import numpy as np

                reply = Image.fromarray(np.asarray(reply))
            except Exception as exc:
                raise TypeError("An image model's call must return the edited image "
                                "(a PIL image or an array) or a GeneratedSegmentation; "
                                f"got {type(reply).__name__}") from exc
        return GeneratedSegmentation(image=reply, provider=provider, model=str(model or ""),
                                     route="custom")

    return call


def image_model(
    source: str | Any,
    *,
    model: str | None = None,
    prompt: str | None = None,
    prompt_file: str | os.PathLike[str] | None = None,
    photograph_first: bool | None = None,
    name: str | None = None,
) -> ImageModel:
    """The image model profile the trace calls.

    *source* is one of:

    - a provider name (``"openai-oauth"``, ``"openai-api"``, ``"gemini-api"``),
      with *model* the image model (None: the
      provider's default). Without *prompt* or *prompt_file* this is the
      provider's built-in profile (LangSlice's prompt; ``tested``).
    - an :class:`~langslice.providers.registry.ImageModel`, or any object
      with a ``call`` attribute (and optionally ``provider`` and ``model``),
      or a plain function: the model of the caller's own. ``call(request)``
      takes a :class:`~langslice.core.nonlinear.types.SegmentationGenerationRequest`
      (``prompt``, ``slice_image`` = Image 1, ``reference_images[0]`` =
      Image 2) and returns the edited image (a PIL image, an array or a
      ``GeneratedSegmentation``).

    *prompt* (text) or *prompt_file* (a UTF-8 text file) is the base prompt
    written for this model; ``{plane}`` in it becomes the section plane.
    *photograph_first* is the attachment order that prompt describes: True,
    Image 1 is the clean photograph and Image 2 the same photograph with the
    placed borders; False, the other way round; None, the order of the
    provider's own prompt (the OpenAI providers photograph first, every
    other provider, a custom one included, borders first). *name* names the
    profile in every trace record (default ``"custom"``).

    A profile with its own prompt or attachment order, or a model of the
    caller's own, is marked untested: each trace it makes records
    ``"untested": true`` and the profile's name. An ``ImageModel`` handed in
    unchanged (no prompt, order or name given) is returned as it is.
    """
    text = _read_prompt(prompt, prompt_file)
    own_inputs = text is not None or photograph_first is not None
    if isinstance(source, str):
        canonical = canonical_provider(source)
        if canonical not in CANONICAL_PROVIDERS or canonical == "none":
            names = [item for item in CANONICAL_PROVIDERS if item != "none"]
            raise ValueError(f"Unknown image-model provider {source!r}; expected one of "
                             f"{names}, or a model object or function of your own")
        resolved = resolve_image_model(canonical, model)
        if not own_inputs:
            return dataclasses.replace(resolved, profile=name or canonical)
        return dataclasses.replace(resolved, prompt=text, photograph_first=photograph_first,
                                   profile=name or CUSTOM_PROFILE, tested=False)
    if isinstance(source, ImageModel):
        if not own_inputs and name is None and model is None:
            return source
        return dataclasses.replace(
            source, model=model or source.model,
            prompt=text if text is not None else source.prompt,
            photograph_first=(photograph_first if photograph_first is not None
                              else source.photograph_first),
            profile=name or (CUSTOM_PROFILE if own_inputs else source.profile),
            tested=source.tested and not own_inputs)
    function = getattr(source, "call", None) or source
    if not callable(function):
        raise TypeError("image_model takes a provider name, an ImageModel, an object with a "
                        f"call, or a function; got {type(source).__name__}")
    declared = canonical_provider(str(getattr(source, "provider", "") or ""))
    provider = declared if declared in CANONICAL_PROVIDERS and declared != "none" \
        else CUSTOM_PROVIDER
    named = model or getattr(source, "model", None)
    return ImageModel(provider=provider, model=None if named is None else str(named),
                      call=_wrapped_call(function, provider, named), prompt=text,
                      photograph_first=photograph_first, profile=name or CUSTOM_PROFILE,
                      tested=False)


__all__ = ["CUSTOM_PROFILE", "default_prompt", "image_model"]
