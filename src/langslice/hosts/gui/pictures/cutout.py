"""A section's picture cut out of its ground, for the 3D view.

The 3D view draws each section as a slab whose picture is cut to its tissue.
The cut is made here, once, from the core's tissue weight
(:func:`langslice.core.sections.tissue.tissue_alpha`): the picture becomes an
RGBA image whose alpha is that weight, trimmed of the dim fringe the mask is
grown over and softened over about a screen pixel, so the browser does no
segmentation. Outside the tissue, and over its trimmed fringe, the colour is
the nearest tissue pixel's, so a viewer that filters the texture linearly
never pulls the ground's colour into the tissue's edge as a halo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    import numpy.typing as npt

ALPHA_MAX = 255


@dataclass(frozen=True)
class EdgeFill:
    """How the cut-out's edge is made.

    The tissue is where the weight is at least *tissue_weight*, trimmed by
    *trim_px* (the mask is grown past the tissue's own edge, over pixels dim
    enough to show as a dark rim) and softened by a Gaussian of *soft_px*
    into the alpha. The pixels deeper than *trim_px* in keep their own colour;
    every other pixel takes the colour of the nearest one that does, blurred
    by a Gaussian of *fill_blur_px* so no streak of one bright pixel runs
    outwards.
    """

    tissue_weight: float = 0.5
    trim_px: float = 8.0
    soft_px: float = 1.2
    fill_blur_px: float = 3.0


EDGE = EdgeFill()


def cut_out(
    painted: Image.Image, weight: npt.NDArray[np.float32], edge: EdgeFill = EDGE
) -> Image.Image:
    """Cut a picture to its tissue.

    Parameters
    ----------
    painted
        The finished picture of a section.
    weight
        The tissue's weight on the same grid, 0 to 1.
    edge
        How the edge is made.

    Returns
    -------
    Image.Image
        RGBA: *painted*'s colour (the ground's replaced by the nearest
        tissue's) with the tissue, trimmed and softened, as its alpha.
    """
    rgb = np.asarray(painted.convert("RGB"))
    tissue = weight >= edge.tissue_weight
    depth = np.asarray(distance_transform_edt(tissue), dtype=np.float32)
    solid = depth >= edge.trim_px
    if not solid.any():
        solid = tissue
    if solid.any():
        nearest = distance_transform_edt(~solid, return_distances=False, return_indices=True)
        rows, columns = np.asarray(nearest, dtype=np.intp)
        blurred = cv2.GaussianBlur(rgb[rows, columns], (0, 0), edge.fill_blur_px)
        rgb = np.where(solid[..., None], rgb, blurred)
    alpha = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), edge.soft_px)
    coverage = np.rint(np.clip(alpha, 0, 1) * ALPHA_MAX).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, coverage]), "RGBA")
