"""The failures LangSlice expects from work that is not its own.

A handler names the exceptions it expects. Two kinds of work fail in more
ways than one call site should list, so each list is stated once here:
reading, writing and drawing images (Pillow, OpenCV, tifffile, numpy and
the files themselves), and running a registration engine (torch, FireANTs,
elastix), which does image work and its own arithmetic. A refusal
(:class:`~langslice.core.refusal.Refused`) is an ``Exception`` and no
``ValueError``, so neither list holds it: a site whose work can refuse and
that means to handle the refusal too names ``Refused`` next to the list.
Anything outside a list is a fault in LangSlice: it is raised on to the one
guard around a tool call, which logs it and answers the call with it.
"""

from __future__ import annotations

import cv2
from PIL import Image

IMAGE_FAILURES: tuple[type[Exception], ...] = (
    ArithmeticError,
    LookupError,
    MemoryError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    cv2.error,
    Image.DecompressionBombError,
)
ENGINE_FAILURES: tuple[type[Exception], ...] = (
    *IMAGE_FAILURES,
    AssertionError,
    AttributeError,
    ImportError,
)
