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
