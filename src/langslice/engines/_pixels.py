"""Pixel work the in-tree engines share: area averaging, the placed atlas
outline, the slide's level, sampling an image at points and outlining a mask.

STalign, emlddmm and SPOmiAlign each read a section on its fit grid
(:class:`~langslice.plugin_api.FitFrame`) and put it, or the atlas, on a grid of
their own; what is the same for all three is written here once.
"""

from __future__ import annotations

import cv2
import numpy as np

from langslice.core.maps.section_maps import point_rows
from langslice.plugin_api import FitFrame

OUTLINE_COLOUR = (230, 30, 160)
OUTLINE_WIDTH_DIVISOR = 600


def area_average(image: np.ndarray, factor: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Average an image down by an area factor, never up.

    Parameters
    ----------
    image
        The image, ``(h, w)`` or ``(h, w, c)``.
    factor
        How many times coarser the result is; a factor of one or less
        returns the image as it is.

    Returns
    -------
    image : numpy.ndarray
        The averaged image.
    scale : tuple of float
        ``(sx, sy)``: down pixel ``k`` sits at original pixel
        ``(k + 0.5) * s - 0.5``.
    """
    h, w = image.shape[:2]
    if factor <= 1.0:
        return image, (1.0, 1.0)
    wd, hd = max(1, round(w / factor)), max(1, round(h / factor))
    return cv2.resize(image, (wd, hd), interpolation=cv2.INTER_AREA), (w / wd, h / hd)


def placed_matrix(frame: FitFrame) -> np.ndarray:
    """Return the map from native atlas pixels to fit-grid pixels as placed.

    Parameters
    ----------
    frame
        The section's fit frame.

    Returns
    -------
    numpy.ndarray
        ``placed_atlas_to_section``, or the frame's own map while the
        section has no transform (3 x 3).
    """
    placed = frame.placed_atlas_to_section
    return np.asarray(frame.atlas_to_section if placed is None else placed, np.float64)


def border_pixels(image: np.ndarray) -> np.ndarray:
    """Return the pixels along the four edges of an image, one row each."""
    return np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])


def slide_level(grey: np.ndarray) -> float:
    """Return the slide's brightness: the median of the image's border."""
    return float(np.median(border_pixels(grey)))


def slide_colour(stain: np.ndarray) -> tuple[float, ...]:
    """Return the slide's colour: the median of the image's border per channel."""
    return tuple(float(v) for v in np.median(border_pixels(stain), axis=0))


def placed_outline(frame: FitFrame, size: tuple[int, int]) -> np.ndarray:
    """Return the placed atlas outline on the fit grid.

    Parameters
    ----------
    frame
        The section's fit frame.
    size
        The fit grid, ``(width, height)``.

    Returns
    -------
    numpy.ndarray
        True where the atlas's annotated plane, at the section's linear
        placement, covers a fit-grid pixel.
    """
    inside = (frame.atlas["annotation"] > 0).astype(np.uint8)
    moved = cv2.warpAffine(inside, placed_matrix(frame)[:2], size, flags=cv2.INTER_NEAREST)
    return moved > 0


def sample_pixels(
    array: np.ndarray,
    points: np.ndarray,
    mode: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Read an image at pixel positions, edges replicated.

    Parameters
    ----------
    array
        The image, ``(h, w)`` or ``(h, w, c)``.
    points
        Where to read it, ``(..., 2)`` as ``(x, y)`` pixels.
    mode
        An OpenCV interpolation flag.

    Returns
    -------
    numpy.ndarray
        The values, shaped like the points without their last axis, with a
        trailing axis of the channels for a 3-D image.
    """
    shape = points.shape[:-1]
    flat = np.asarray(points, np.float32).reshape(-1, 2)
    xs, ys = point_rows(flat[:, 0]).astype(np.float32), point_rows(flat[:, 1]).astype(np.float32)
    channels = [array] if array.ndim == 2 else [array[..., c] for c in range(array.shape[2])]
    read = [
        cv2.remap(
            np.ascontiguousarray(plane, dtype=np.float32),
            xs,
            ys,
            mode,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)[: len(flat)]
        for plane in channels
    ]
    if array.ndim == 3:
        return np.stack(read, -1).reshape(*shape, len(read))
    return read[0].reshape(shape)


def outlined(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of an RGB image with the outlines of a mask drawn on it.

    Parameters
    ----------
    image
        The image, ``(h, w, 3)`` uint8.
    mask
        The region to outline, ``(h, w)``.

    Returns
    -------
    numpy.ndarray
        The image with every outline of the mask drawn in the engines'
        outline colour, a line of about a six-hundredth of the width.
    """
    drawn = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    width = max(1, image.shape[1] // OUTLINE_WIDTH_DIVISOR)
    cv2.drawContours(drawn, contours, -1, OUTLINE_COLOUR, width)
    return drawn
