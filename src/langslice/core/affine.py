"""In-plane affines: the matrix builders and the normalized six-number convention.

Shared by every reader of a stored transform. Pure functions only — no CLI,
no agent, no model calls, no file writes. :func:`affine_matrix` builds the
2x3 from human-readable knobs (rotation, per-axis scale, translation), which
is what the interactive transform proposes; :func:`decompose_affine` reads
the knobs back.

PARAMETER CONVENTION. Matrices are OpenCV 2x3 row-major, in PIXELS of a
working frame::

    [x_dst]   [a  b  tx] [x_src]
    [y_dst] = [c  d  ty] [y_src]
                         [  1  ]

:func:`normalized_affine` converts one to the resolution-independent 6-vector
``[a, b, tx, c, d, ty]`` that gets handed back to hosts, where x is a fraction
of image width and y a fraction of image height, so applying it to the user's
full-resolution section needs nothing but the section's own size.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: The identity's six normalized numbers (:func:`normalized_affine`): what a
#: section without a transform is drawn and mapped with.
IDENTITY_PARAMS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def pixel_center_map(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """3x3 map of pixel centres from one grid onto a resized (and shifted) grid.

    Pixel centres sit at integer indices (OpenCV's convention), so a resize
    by ``s`` maps ``x`` to ``s * x + (s - 1) / 2``; each axis keeps its own
    factor (resize rounding), and *offset* adds padding or a placement on a
    larger canvas, in target pixels. The one copy: the image tool's canvas,
    the nonlinear canvas and the deformable working grid all use it.
    """
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    return np.array([
        [sx, 0.0, offset[0] + (sx - 1.0) / 2.0],
        [0.0, sy, offset[1] + (sy - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """Scale *image* so its long edge is exactly *long_edge* px."""
    width, height = image.size
    current = max(width, height)
    if current == long_edge:
        return image
    scale = long_edge / float(current)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )


def affine_matrix(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x: float,
    translate_y: float,
    size: tuple[int, int],
    pivot: tuple[float, float] | None = None,
    shear: float = 0.0,
) -> np.ndarray:
    """A 2x3 affine from human knobs, about the centre of a *size* image.

    Rotation is counter-clockwise on screen (the OpenCV convention), scales
    are multipliers per axis applied before the rotation, and translations are
    FRACTIONS of image width/height — positive x moves right, positive y moves
    down. Identity is ``rotation_deg=0, scale=1, translate=0, shear=0``.

    *shear* is :func:`decompose_affine`'s: the linear part is
    ``R(rotation) . [[scale_x, shear * scale_x], [0, scale_y]]``, so before
    the rotation a point ``(x, y)`` (from the pivot) moves sideways by
    ``shear * y``, in units of ``scale_x``: a dimensionless slant, ``0.1``
    shifting each row by a tenth of its distance below the pivot. The knobs
    and :func:`decompose_affine` round-trip exactly (up to its rounding).

    *pivot* moves the point rotation and scale happen about, in PIXELS of the
    same frame *size* describes; ``None`` is the frame's centre. The
    translation is unaffected — it is applied after, whatever the pivot.
    """
    width, height = size
    cx, cy = (width / 2.0, height / 2.0) if pivot is None else (float(pivot[0]), float(pivot[1]))
    rad = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    a, b = cos_t * scale_x, sin_t * scale_y
    c, d = -sin_t * scale_x, cos_t * scale_y
    if shear:
        b += cos_t * shear * scale_x
        d -= sin_t * shear * scale_x
    return np.array([
        [a, b, cx - (a * cx + b * cy) + translate_x * width],
        [c, d, cy - (c * cx + d * cy) + translate_y * height],
    ], dtype=np.float64)


def physical_affine_matrix(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x_mm: float,
    translate_y_mm: float,
    size: tuple[int, int],
    um_per_px: float,
    pivot: tuple[float, float] | None = None,
    shear: float = 0.0,
) -> np.ndarray:
    """:func:`affine_matrix` with the shifts given in MILLIMETRES.

    The registration-software convention (ABBA's): rotation in degrees about
    the frame's centre, unitless per-axis scales, translations in mm on a
    frame whose pixels are *um_per_px* micrometres wide. One millimetre is
    ``1000 / um_per_px`` pixels, whatever the frame's size — which is the
    whole point of calibrating: the same numbers mean the same displacement
    at any working resolution.

    *pivot* is the rotation/scale centre in pixels of *size*; ``None`` is the
    frame's centre. Since the frame's width and height cancel out of the
    translation, the SAME numbers build the same map on the section's frame
    and on the padded canvas around it — as long as the pivot is expressed in
    whichever frame *size* names.
    """
    if um_per_px <= 0:
        raise ValueError("um_per_px must be positive")
    px_per_mm = 1000.0 / float(um_per_px)
    width, height = size
    return affine_matrix(
        rotation_deg=rotation_deg,
        scale_x=scale_x,
        scale_y=scale_y,
        translate_x=translate_x_mm * px_per_mm / width,
        translate_y=translate_y_mm * px_per_mm / height,
        size=size,
        pivot=pivot,
        shear=shear,
    )


def normalized_physical_affine(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x_mm: float,
    translate_y_mm: float,
    size: tuple[int, int],
    um_per_px: float,
    pivot: tuple[float, float] | None = None,
    shear: float = 0.0,
) -> list[float]:
    """Physical parameters as the six normalized numbers hosts consume.

    *size* is the SECTION's own frame, not the padded canvas the parameters
    were chosen on: the pad is symmetric, so both frames share a centre, and
    a rotation about the canvas centre is the same map as a rotation about
    the section centre. Only the frame the numbers are expressed in changes.

    That equivalence is exactly what a *pivot* breaks, so a pivot chosen on
    the canvas must be handed over in the SECTION's frame (subtract the
    canvas's section offset). The six numbers then describe the same map the
    canvas showed, still on the section's own frame.
    """
    return normalized_affine(
        physical_affine_matrix(
            rotation_deg=rotation_deg,
            scale_x=scale_x,
            scale_y=scale_y,
            translate_x_mm=translate_x_mm,
            translate_y_mm=translate_y_mm,
            size=size,
            um_per_px=um_per_px,
            pivot=pivot,
            shear=shear,
        ),
        size,
    )


def decompose_affine(
    params_or_matrix: Any, size: tuple[int, int] | None = None
) -> dict[str, Any]:
    """A 2x3 affine as the numbers a human reads: rotation, scale, shear, shift.

    Accepts either the normalized 6-vector ``[a, b, tx, c, d, ty]`` or a 2x3
    matrix. The decomposition is ``M = R(rotation) . [[sx, shear*sx], [0, sy]]``
    read back out by QR: ``scale_x`` is the length of the first column,
    ``rotation_deg`` its angle (counter-clockwise on screen, the
    :func:`affine_matrix` convention), ``shear`` the residual x/y coupling in
    units of ``scale_x``, and ``scale_y`` the determinant over ``scale_x`` —
    so it goes NEGATIVE exactly when the map includes a reflection.
    ``mirrored`` reports that determinant sign on its own.

    A normalized 6-vector lives in a fractional frame (x/width, y/height),
    which is anisotropic on a non-square image: a pure 5-degree rotation reads
    as 7 degrees plus shear there. Pass *size* to decompose in the pixel frame
    instead — angles and scales then mean what they say.

    Translations come back as the input's own ``tx``/``ty``, which for the
    normalized 6-vector every payload passes are FRACTIONS of width and
    height. Hence the names: a 0.15 mm shift on a 20 mm-wide frame reads as
    ``translate_x_frac = 0.0075``, and reading that as millimetres is exactly
    the confusion the suffix exists to stop.
    """
    values = np.asarray(params_or_matrix, dtype=np.float64).reshape(2, 3)
    (a, b, tx), (c, d, ty) = values[0], values[1]
    if size is not None:
        width, height = float(size[0]), float(size[1])
        b, c = b * width / height, c * height / width
    scale_x = float(math.hypot(a, c))
    determinant = float(a * d - b * c)
    rotation = math.degrees(math.atan2(-c, a)) if scale_x > 0 else 0.0
    shear = float((a * b + c * d) / (scale_x**2)) if scale_x > 0 else 0.0
    scale_y = determinant / scale_x if scale_x > 0 else 0.0
    return {
        "rotation_deg": round(rotation, 3),
        "scale_x": round(scale_x, 4),
        "scale_y": round(scale_y, 4),
        "shear": round(shear, 4),
        "translate_x_frac": round(float(tx), 4),
        "translate_y_frac": round(float(ty), 4),
        "mirrored": determinant < 0,
    }


def denormalized_affine(params: Any, size: tuple[int, int]) -> np.ndarray:
    """The inverse of :func:`normalized_affine`: six numbers back to pixels.

    What a stored transform has to go through to be DRAWN again — the six
    numbers are exact, where the reported knobs (shear included) are
    rounded, so a picture of "what is stored" is built from these and not
    from the knobs.
    """
    a, b, tx, c, d, ty = (float(v) for v in params)
    width, height = float(size[0]), float(size[1])
    return np.array(
        [[a, b * width / height, tx * width], [c * height / width, d, ty * height]],
        dtype=np.float64,
    )


def normalized_affine(matrix: np.ndarray, size: tuple[int, int]) -> list[float]:
    """A pixel-space 2x3 as the resolution-independent ``[a, b, tx, c, d, ty]``.

    Both source and destination are expressed in fractions of *size*, so the
    result can be applied to the section at any resolution. Source and
    destination frames are assumed to share *size* — which they do, because
    the atlas mask is built at the resized section's size.
    """
    width, height = size
    (a, b, tx), (c, d, ty) = matrix[0], matrix[1]
    return [
        float(a),
        float(b) * height / width,
        float(tx) / width,
        float(c) * width / height,
        float(d),
        float(ty) / height,
    ]
