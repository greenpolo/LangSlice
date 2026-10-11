"""Exact affine pixel geometry carried alongside unannotated renderer images."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from PIL import Image

KEYS = ("pixel_to_file_fraction", "pixel_to_native")


def cropped_geometry(
    image: Image.Image, source: Image.Image, box: tuple[int, int, int, int] | None = None
) -> Image.Image:
    """Carry source pixel-centre maps through PIL crop/resize operations."""
    x0, y0, x1, y1 = box or (0, 0, source.width, source.height)
    sx, sy = (x1 - x0) / image.width, (y1 - y0) / image.height
    to_source = np.array([[sx, 0, x0 + (sx - 1) / 2], [0, sy, y0 + (sy - 1) / 2], [0, 0, 1.0]])
    for key in KEYS:
        held: Any = source.info.get(key)
        if held is not None:
            image.info[key] = (np.asarray(held) @ to_source).tolist()
    return image


def resized_recipe(recipe: dict[str, Any] | None, source: tuple[int, int],
                   target: tuple[int, int]) -> dict[str, Any] | None:
    """Saved-picture geometry after a door resizes the entire captioned image.

    Look recipes remain redrawable at native detail. Other recipes become
    immutable snapshots; their next crop starts from the delivered image.
    """
    if recipe is None:
        return None
    out = copy.deepcopy(recipe)
    sx, sy = source[0] / target[0], source[1] / target[1]
    to_source = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
    for panel in out.get("endpoint_panels", []):
        panel["bounds"] = (np.asarray(panel["bounds"]) / [sx, sy, sx, sy]).tolist()
        for key in (*KEYS, "pixel_to_render"):
            if key in panel:
                panel[key] = (np.asarray(panel[key]) @ to_source).tolist()
    if out.get("shown"):
        out["shown"] = (np.asarray(out["shown"]) / [sx, sy]).tolist()
    if out.get("renderer") != "look":
        out["renderer"] = "transport_snapshot"
        out["args"] = {}
        out.pop("source_endpoint_panels", None)
    return out


# --- the source picture's pixel system a zoom is read in -------------------------------

#: Round grid steps, in source-picture pixels.
GRID_STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000)
#: At most this many grid lines along a zoom's longer side.
GRID_LINES = 12


class SourceFrame:
    """Where a zoom's pixels lie in the pixels of the whole (unzoomed) picture
    it was cut from: ``source = origin + (pixel + 0.5) * scale - 0.5`` per
    axis (pixel centres at integers, as every picture coordinate here)."""

    def __init__(self, origin: Any, scale: Any, shown: Any) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.shown = np.asarray(shown, dtype=float)

    def to_source(self, xy: Any) -> np.ndarray:
        return self.origin + (np.asarray(xy, dtype=float) + 0.5) * self.scale - 0.5

    def to_picture(self, xy: Any) -> np.ndarray:
        return (np.asarray(xy, dtype=float) + 0.5 - self.origin) / self.scale - 0.5

    def edge_to_picture(self, xy: Any) -> np.ndarray:
        """A box corner (a pixel EDGE: the left edge of pixel 0 is 0) in the
        whole picture's numbers, as the zoom's own edge coordinate."""
        return (np.asarray(xy, dtype=float) - self.origin) / self.scale

    def covers(self) -> list[float]:
        """``[x0, y0, x1, y1]``: the source pixels the zoom's content spans."""
        low, high = self.to_source([0, 0]), self.to_source(self.shown - 1)
        return [float(v) for v in (*low, *high)]


def source_frame(recipe: dict[str, Any] | None) -> SourceFrame | None:
    """The :class:`SourceFrame` of a zoomed picture's *recipe* (its window on
    the unzoomed picture, else the box a crop was cut at); None for a picture
    that is itself whole, whose pixels are its own."""
    if not recipe:
        return None
    shown = recipe.get("shown")
    args = recipe.get("args") or {}
    if not shown or min(shown) <= 0:
        return None
    if recipe.get("renderer") == "crop":
        box = args.get("box")
        if not box:
            return None
        low, high = np.asarray(box[:2], dtype=float), np.asarray(box[2:], dtype=float)
    else:
        window, base = args.get("zoom"), recipe.get("base")
        if not window or not base:
            return None
        low = np.asarray(window[:2], dtype=float) * base
        high = np.asarray(window[2:], dtype=float) * base
    scale = (high - low) / np.asarray(shown, dtype=float)
    return SourceFrame(low, scale, shown)


def grid_step(extent: float) -> int:
    """The round step (pixels) that puts at most :data:`GRID_LINES` lines along
    *extent* source pixels, and so about six or more."""
    for step in GRID_STEPS:
        if extent / step <= GRID_LINES:
            return step
    return GRID_STEPS[-1]
