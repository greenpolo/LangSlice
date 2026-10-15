"""Which way is up on the image model's canvas: the section turned upright.

An image model draws a coronal section's two hemispheres far less
symmetrically when the section is tilted on its canvas, so the canvas shows
the section turned so the atlas stands upright, from its linear placement
(:func:`upright`): the placed atlas plane's vertical points straight up, and
on a coronal or horizontal section the atlas midline runs down the middle of
the canvas. It is a turn and a shift, never a mirror: a flip the placement
holds stays. :func:`~langslice.core.nonlinear.image_gen_registration.prepare_canvas`
draws it; the turn is part of the atlas-to-canvas matrix, so the lines the
model draws are read back through it exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from langslice.core.space import Plane


@dataclass(frozen=True)
class Upright:
    """How a section render stands on the canvas.

    *up* is a direction in the render's pixels (x right, y down) that points
    straight up on the canvas; *through* is a render point (pixel centres at
    integers) the canvas's vertical centre line passes through, or None to
    centre the turned render. The default leaves the render as it is.
    """

    up: tuple[float, float] = (0.0, -1.0)
    through: tuple[float, float] | None = None

    @property
    def turn_deg(self) -> float:
        """The turn, in degrees clockwise on the screen, in (-180, 180]."""
        ux, uy = self.up
        # The turn R(phi) takes `up` to (0, -1): phi = -90 deg - angle(up).
        phi = -90.0 - math.degrees(math.atan2(uy, ux))
        phi = (phi + 180.0) % 360.0 - 180.0
        return 180.0 if phi == -180.0 else phi

    def rotation(self) -> np.ndarray:
        """2x2 turn on render pixels (x right, y down) that points *up* upward."""
        phi = math.radians(self.turn_deg)
        c, s = math.cos(phi), math.sin(phi)
        return np.array([[c, -s], [s, c]])


def upright(
    atlas: Any,
    plane: Plane,
    position_mm: float,
    pitch_deg: float,
    yaw_deg: float,
    atlas_to_render: np.ndarray,
) -> Upright:
    """The upright stance of a placed section render.

    *atlas_to_render* is the linear placement (3x3, native atlas-plane pixel
    centres ``[x, y, 1]`` to render pixel centres). Up on the canvas is,
    carried through the placement:

    - coronal: the atlas midline (the plane's line through the medio-lateral
      centre of the volume, where BrainGlobe splits the hemispheres), its
      dorsal end up, the canvas centred on it;
    - horizontal: the atlas midline, its anterior end up, centred on it;
    - sagittal: the plane's dorsal direction (no midline to centre on).

    Cutting angles are the section's own, so the midline is the line where
    its oblique plane crosses the volume's midplane.
    """
    from langslice.core.oblique import plane_index_affine
    from langslice.core.space import atlas_space_context

    context = atlas_space_context(atlas)
    # [row, col, 1] -> volume index per axis; gradients below are d index / d (x, y).
    index = plane_index_affine(atlas, position_mm, plane, pitch_deg, yaw_deg)

    def gradient(axis: int) -> np.ndarray:
        return np.array([index[axis, 1], index[axis, 0]], dtype=np.float64)

    dv = context.dv_axis_index
    # Toward dorsal on the native plane: down the DV index where index 0 is superior.
    dorsal = gradient(dv) * (-1.0 if context.space.origin[dv] == "s" else 1.0)
    linear = np.asarray(atlas_to_render, dtype=np.float64)[:2, :2]
    if plane == "sagittal":
        return Upright(up=_carried(linear, dorsal))

    ml = context.ml_axis_index
    across = gradient(ml)
    if np.hypot(*across) < 1e-9:  # the plane runs along the midplane: no midline
        return Upright()
    along = np.array([-across[1], across[0]])
    # Coronal: the dorsal end up. Horizontal: the anterior end up (AP index 0
    # is anterior on every supported atlas, `space.atlas_space_context`).
    aim = dorsal if plane == "coronal" else -gradient(context.ap_axis_index)
    if float(along @ aim) < 0:
        along = -along
    # The midline point nearest the plane's centre: ML index there is the split.
    count = context.shape[ml]
    split = round(count / 2) - 0.5
    rows, cols = _plane_shape(context, plane)
    centre = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])
    at_centre = index[ml, 0] * centre[1] + index[ml, 1] * centre[0] + index[ml, 2]
    point = centre + (split - at_centre) * across / float(across @ across)
    shown = np.asarray(atlas_to_render, dtype=np.float64) @ np.array([*point, 1.0])
    return Upright(up=_carried(linear, along), through=(float(shown[0]), float(shown[1])))


def _carried(linear: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    """A native-plane direction as it points on the render."""
    vector = linear @ direction
    return float(vector[0]), float(vector[1])


def _plane_shape(context: Any, plane: Plane) -> tuple[int, int]:
    """(rows, cols) of the display-oriented native plane."""
    from langslice.core.space import slice_axis_index

    normal = slice_axis_index(context, plane)
    row, col = (context.shape[a] for a in range(3) if a != normal)
    return (col, row) if plane in ("sagittal", "horizontal") else (row, col)
