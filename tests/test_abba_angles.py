"""ABBA's slicing rotations <-> LangSlice pitch/yaw (no JVM required).

The sign table is :mod:`langslice.core.abba_angles`; the Fiji connector
applies stack-wide angle changes with it (``host_angles``: rotateX =
-radians(pitch_deg), rotateY = -radians(yaw_deg)).
"""

import numpy as np
import pytest

# --- ABBA slicing rotations <-> LangSlice pitch/yaw ---------------------------

#: Stage 0 (atlas (ML, DV, AP) mm -> ABBA world) of ABBA's own QuPath
#: ``ABBA-Transform-allen_mouse_10um_java.json`` exports, with the save's
#: rotationX/rotationY in radians. Only the 3x3 block;
#: world z is the slicing axis, so its row is the cutting plane's normal.
#: ABBA's ASR atlas names x > 5.7 "Left", BrainGlobe asr names the high ML index
#: left, and the exported Left ROIs sit on the image's right for every
#: unflipped section , so x IS BrainGlobe ML here.
_ABBA_EXPORTS = {
    "export_1": (
        0.22689280275926282,
        0.03490658503988659,
        [
            [1.0006095442988217, -0.008062094885272547, 0.0],
            [0.0, 1.0263041077933914, 0.0],
            [0.03492076949174772, -0.23100891551524302, 0.9999999999999999],
        ],
    ),
    "export_2": (
        -0.148352986419518,
        0.05235987755982988,
        [
            [1.0013723459979211, 0.00783239509233458, 0.0],
            [0.0, 1.0111061278640618, 0.0],
            [0.052407779283041175, 0.14965609983271452, 1.0],
        ],
    ),
}


def _langslice_normal_ml_dv_ap(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """LangSlice's coronal cutting-plane normal on an asr atlas, as (ML, DV, AP)."""
    from langslice.core.oblique import build_rotation_matrix

    # asr: axis 0 = AP (the coronal normal), 1 = DV (rows), 2 = ML (columns)
    n = build_rotation_matrix(pitch_deg, yaw_deg, row_axis=1, col_axis=2) @ np.array(
        [1.0, 0.0, 0.0]
    )
    return np.array([n[2], n[1], n[0]])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = abs(float(a @ b)) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, cos))))


@pytest.mark.parametrize("section", sorted(_ABBA_EXPORTS))
def test_rotate_sign_constants_reproduce_abba_exported_plane(section):
    import math

    from langslice.core.abba_angles import (
        PITCH_TO_ROTATE_X_SIGN,
        YAW_TO_ROTATE_Y_SIGN,
        angles_to_rotate,
        rotate_to_angles,
    )

    rx, ry, stage0 = _ABBA_EXPORTS[section]
    abba_normal = np.asarray(stage0)[2]
    # the read-back direction (angle = deg(rotate) / sign)
    pitch, yaw = rotate_to_angles(rx, ry)
    assert pitch == pytest.approx(math.degrees(rx) / PITCH_TO_ROTATE_X_SIGN)
    assert yaw == pytest.approx(math.degrees(ry) / YAW_TO_ROTATE_Y_SIGN)
    assert _angle_deg(_langslice_normal_ml_dv_ap(pitch, yaw), abba_normal) < 1e-6
    # the opposite yaw sign is measurably wrong (2-3 deg of yaw -> 4-6 deg off)
    assert _angle_deg(_langslice_normal_ml_dv_ap(pitch, -yaw), abba_normal) > 3.0
    # and the set direction is the exact inverse of the read-back
    assert angles_to_rotate(pitch, yaw) == pytest.approx((rx, ry))


def test_backward_compatible_module_path_reexports_the_signs():
    """SliceBench reads the two constants from the old module path."""
    from langslice.core import abba_angles
    from langslice.integrations.abba_linear import (
        PITCH_TO_ROTATE_X_SIGN,
        YAW_TO_ROTATE_Y_SIGN,
    )

    assert PITCH_TO_ROTATE_X_SIGN == abba_angles.PITCH_TO_ROTATE_X_SIGN == -1.0
    assert YAW_TO_ROTATE_Y_SIGN == abba_angles.YAW_TO_ROTATE_Y_SIGN == -1.0
