"""The golden stack: three real sections and the fixed inputs every door uses.

Each door's scenario (``library_door.py``, ``cli_door.py``,
``mcp_door.py``) imports these, so the doors register the same sections with
the same numbers and their outputs can be read side by side.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

FRONT = "M01_001_005.png"
MIDDLE = "M01_002_005.png"
BACK = "M01_002_009.png"
SECTIONS = (FRONT, MIDDLE, BACK)

ATLAS = "allen_mouse_50um"
TASKS = ("position", "transform", "nonlinear")
PIXEL_SIZE_UM = 12.8
POSITIONS_MM = {FRONT: 3.5, MIDDLE: 5.62, BACK: 6.29}
CUTTING_ANGLES = {"pitch_deg": 1.5, "yaw_deg": -1.0}
HAND_TRANSFORMS = [
    {"id": FRONT, "rotation_deg": 2.0, "scale_x": 1.05, "translate_x_mm": 0.1},
    {"id": MIDDLE, "scale_y": 0.97, "shear": 0.02},
]
ZOOM_BOX = [120, 80, 380, 260]
MISSING = "missing.png"


def placements() -> list[dict[str, object]]:
    """Return every section's position, as ``position_sections`` takes it.

    Returns
    -------
    list of dict
        ``{id, position_mm}`` per section, in stack order.
    """
    return [
        {"id": name, "position_mm": POSITIONS_MM[name]} for name in SECTIONS
    ]


def picture_digest(image: Image.Image) -> dict[str, object]:
    """Return the size and pixel digest of one picture a reply carried.

    Parameters
    ----------
    image
        A decoded picture: a library reply's image before JPEG, or an MCP
        image block decoded.

    Returns
    -------
    dict
        ``size`` as ``[width, height]`` and ``sha256`` of its RGB pixels.
    """
    pixels = np.asarray(image.convert("RGB"))
    return {
        "size": [pixels.shape[1], pixels.shape[0]],
        "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
