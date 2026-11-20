"""Provider names and image-model resolution.

A provider is an ACCESS METHOD, a vendor plus how you authenticate. What each
one is (label, credential, models, prices) is the table in
:mod:`langslice.providers.table`; the names the core compares are in
:mod:`langslice.core.connections` (re-exported here). ``none`` means no image
model: FireANTs fits to the stain directly, with nothing to authenticate. A new
provider is a row in the table and a transport in this package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from langslice.core.connections import (
    CANONICAL_PROVIDERS,
    CUSTOM_PROVIDER,
    canonical_provider,
)
from langslice.core.fits.trace.request import ImageModel
from langslice.providers.table import find

if TYPE_CHECKING:
    from langslice.core.fits.trace.request import (
        BorderTraceRequest,
        GeneratedSegmentation,
    )


def openai_oauth_credentials_path() -> Path:
    """The ``openai-oauth`` login file (``langslice login`` writes it, mode
    600): ``LANGSLICE_OPENAI_AUTH`` when set (another account for one
    process), else ``~/.langslice/openai_auth.json``. Read at call time."""
    override = os.environ.get("LANGSLICE_OPENAI_AUTH", "").strip()
    return (
        Path(override).expanduser()
        if override
        else (Path.home() / ".langslice" / "openai_auth.json")
    )


def default_image_model(provider: str) -> str | None:
    """The image model a provider uses when none is named (None for
    ``openai-api``: its transport reads ``OPENAI_IMAGE_MODEL``)."""
    found = find(provider)
    return None if found is None else found.default_image_model()


def resolve_image_model(provider: str, model: str | None = None) -> ImageModel:
    """Resolve a provider name to its image-edit call.

    The transport (``langslice.providers.image_calls``) is imported when the
    call runs, not here, so resolving loads no model client. ``none`` and
    unknown names are refused: there is no image model to call.
    """
    canonical = canonical_provider(provider)
    if canonical == "none":
        raise ValueError("The image correction tool requires an image-model provider")
    if canonical == CUSTOM_PROVIDER:
        raise ValueError(
            "This job's image model is the caller's own (provider 'custom'): "
            "hand it to the library, langslice.open_job(..., image_model=...)"
        )
    if canonical not in CANONICAL_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    def call(request: BorderTraceRequest) -> GeneratedSegmentation:
        from langslice.providers.image_calls import (
            generate_warped_segmentation_image,
        )

        return generate_warped_segmentation_image(request)

    return ImageModel(provider=canonical, model=model or default_image_model(canonical), call=call)


__all__ = [
    "CANONICAL_PROVIDERS",
    "CUSTOM_PROVIDER",
    "canonical_provider",
    "default_image_model",
    "openai_oauth_credentials_path",
    "resolve_image_model",
]
