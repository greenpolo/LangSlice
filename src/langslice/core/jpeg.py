"""The one JPEG encoding every picture a model receives goes through.

The doors send pictures as JPEG (the ADK agent's message parts,
:mod:`langslice.adk.media`; MCP image blocks); the job saves the same bytes
(:class:`langslice.job.views.ViewStore`) by calling the same function on the
same picture. Pure Pillow, deterministic for a given picture.
"""

from __future__ import annotations

import io

from PIL import Image

#: JPEG quality of every picture a tool or the opening sends.
JPEG_QUALITY = 85


def encode_jpeg(img: Image.Image, *, quality: int = JPEG_QUALITY) -> bytes:
    """One PIL image as JPEG bytes (RGB), the encoding every door sends."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
