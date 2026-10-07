"""ABBA's atlas slicing rotations <-> LangSlice's cutting angles.

ABBA (0.24.x) tilts the atlas for the WHOLE stack with
``ReslicedAtlas.setRotateX`` / ``setRotateY`` (radians); LangSlice states a
plane as ``pitch_deg`` / ``yaw_deg``. The mapping is a sign per axis::

    rotateX = radians(pitch_deg * PITCH_TO_ROTATE_X_SIGN)
    rotateY = radians(yaw_deg * YAW_TO_ROTATE_Y_SIGN)

Both signs are -1, pinned against stage 0 of ABBA's own exported transform
chains (``tests/test_abba_angles.py``). A +rotateY cut puts the screen-LEFT
half more posterior; LangSlice's coronal display has the BrainGlobe ML index
running rightward, so that is yaw < 0. ABBA's ML coordinate channel
DECREASES with screen x, and the left-right symmetric Allen labels cannot
reveal a mirror, so an ML sign is never settled by label agreement. The
Fiji connector applies the same mapping (``SlicingAnglesAction``).
"""

from __future__ import annotations

import math

#: Which of ReslicedAtlas.setRotateX/setRotateY is pitch vs yaw, and sign.
#: pitch_deg = -deg(rotateX), yaw_deg = -deg(rotateY).
PITCH_TO_ROTATE_X_SIGN = -1.0
YAW_TO_ROTATE_Y_SIGN = -1.0


def angles_to_rotate(pitch_deg: float, yaw_deg: float) -> tuple[float, float]:
    """``(rotateX, rotateY)`` in radians for ABBA's ``ReslicedAtlas``."""
    return (math.radians(float(pitch_deg) * PITCH_TO_ROTATE_X_SIGN),
            math.radians(float(yaw_deg) * YAW_TO_ROTATE_Y_SIGN))


def rotate_to_angles(rotate_x: float, rotate_y: float) -> tuple[float, float]:
    """``(pitch_deg, yaw_deg)`` of ABBA's ``ReslicedAtlas`` rotations (radians)."""
    return (math.degrees(float(rotate_x)) / PITCH_TO_ROTATE_X_SIGN,
            math.degrees(float(rotate_y)) / YAW_TO_ROTATE_Y_SIGN)
