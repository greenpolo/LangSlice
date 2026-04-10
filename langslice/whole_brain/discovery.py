"""Discover and naturally sort slice images in a folder."""

from __future__ import annotations

import os
import re

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: str) -> list[str | int]:
    """Sort key that orders embedded numbers numerically."""
    basename = os.path.basename(path).lower()
    parts: list[str | int] = []
    for piece in _NATURAL_SORT_RE.split(basename):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            parts.append(piece)
    return parts


def discover_slices(folder: str) -> list[str]:
    """Return absolute paths to slice images in *folder*, naturally sorted.

    Scans for files with extensions: .png, .jpg, .jpeg, .tif, .tiff.
    Non-image files are silently skipped.
    """
    hits: list[str] = []
    for entry in os.listdir(folder):
        ext = os.path.splitext(entry)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            hits.append(os.path.join(folder, entry))
    hits.sort(key=_natural_sort_key)
    return hits
