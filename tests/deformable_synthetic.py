"""A tiny synthetic atlas and sections deformed by a KNOWN residual field.

The atlas is six identical coronal planes (50 um voxels) of nested ellipses
with a structure tree, colors and a textured template. A section is made by
pushing the linearly placed atlas through a chosen field ``u`` in the
documented convention: section pixel p shows the placed-atlas point
``p + u(p)`` (u in millimetres), rendered as a brightfield stain.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from langslice.core.deformable.geometry import Placement

CTX, STR, TH, HY, VS, VL = 2, 3, 4, 5, 73, 81
SECTION_MM_PER_PX = 0.025
ATLAS_TO_SECTION = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 8.0], [0.0, 0.0, 1.0]])
SECTION_SIZE = (340, 260)


class SyntheticAtlas:
    atlas_name = "synthetic_ellipses_50um"
    orientation = "asr"
    resolution = (50.0, 50.0, 50.0)
    metadata = {"species": "mouse"}

    def __init__(self) -> None:
        plane = np.zeros((120, 160), dtype=np.int32)
        cv2.ellipse(plane, (80, 62), (70, 52), 0, 0, 360, CTX, -1)
        cv2.ellipse(plane, (52, 58), (22, 16), 0, 0, 360, STR, -1)
        cv2.ellipse(plane, (108, 62), (20, 18), 0, 0, 360, TH, -1)
        cv2.ellipse(plane, (80, 98), (24, 9), 0, 0, 360, HY, -1)
        cv2.ellipse(plane, (54, 38), (9, 4), 0, 0, 360, VL, -1)
        gray = {0: 0.0, CTX: 110.0, STR: 190.0, TH: 55.0, HY: 160.0, VL: 8.0}
        template = np.zeros(plane.shape, dtype=np.float32)
        for uid, value in gray.items():
            template[plane == uid] = value
        yy, xx = np.indices(plane.shape)
        texture = 18.0 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
        template = np.where(plane > 0, np.clip(template + texture * (plane != VL), 0, 255), 0)
        template = cv2.GaussianBlur(template.astype(np.float32), (0, 0), 0.7)
        self.annotation = np.repeat(plane[None], 6, axis=0)
        self.template = np.repeat(template[None], 6, axis=0).astype(np.float32)
        self.reference = self.template
        colors = {1: (200, 200, 200), CTX: (50, 160, 60), STR: (120, 160, 220),
                  TH: (240, 120, 120), HY: (230, 80, 200), VS: (170, 170, 170),
                  VL: (150, 150, 150)}
        paths = {1: [1], CTX: [1, CTX], STR: [1, STR], TH: [1, TH], HY: [1, HY],
                 VS: [1, VS], VL: [1, VS, VL]}
        acronyms = {1: "root", CTX: "CTX", STR: "STR", TH: "TH", HY: "HY", VS: "VS", VL: "VL"}
        self.structures = {
            uid: {"id": uid, "acronym": acronyms[uid], "name": acronyms[uid],
                  "structure_id_path": paths[uid], "rgb_triplet": list(colors[uid])}
            for uid in paths
        }


def placement() -> Placement:
    return Placement(
        atlas_name=SyntheticAtlas.atlas_name, position_mm=0.1, plane="coronal",
        pitch_deg=0.0, yaw_deg=0.0, atlas_to_section=ATLAS_TO_SECTION,
        section_mm_per_px=SECTION_MM_PER_PX,
    )


def bump_field(
    centres: list[tuple[float, float, float, float, float]],
    size: tuple[int, int] = SECTION_SIZE,
) -> np.ndarray:
    """Sum of Gaussian bumps (cx, cy, sigma_px, dx_mm, dy_mm) on the section grid."""
    width, height = size
    yy, xx = np.indices((height, width), dtype=np.float64)
    field = np.zeros((height, width, 2))
    for cx, cy, sigma, dx, dy in centres:
        weight = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        field[..., 0] += dx * weight
        field[..., 1] += dy * weight
    return field


SMOOTH_FIELD: Callable[[], np.ndarray] = lambda: bump_field(  # noqa: E731
    [(130, 120, 45, 0.09, 0.03), (230, 140, 40, -0.05, -0.07)])


def render_section(
    atlas: SyntheticAtlas, field_mm: np.ndarray, *, remove: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[Image.Image, np.ndarray]:
    """(brightfield RGB section, its true leaf labels) under *field_mm*.

    *remove* blanks section pixels to slide background (missing tissue).
    """
    width, height = SECTION_SIZE
    yy, xx = np.indices((height, width), dtype=np.float64)
    qx = xx + field_mm[..., 0] / SECTION_MM_PER_PX
    qy = yy + field_mm[..., 1] / SECTION_MM_PER_PX
    inverse = np.linalg.inv(ATLAS_TO_SECTION)
    nx = (inverse[0, 0] * qx + inverse[0, 1] * qy + inverse[0, 2]).astype(np.float32)
    ny = (inverse[1, 0] * qx + inverse[1, 1] * qy + inverse[1, 2]).astype(np.float32)
    template = atlas.template[0]
    labels_native = atlas.annotation[0].astype(np.float32)
    values = cv2.remap(template, nx, ny, cv2.INTER_LINEAR, borderValue=0)
    labels = cv2.remap(labels_native, nx, ny, cv2.INTER_NEAREST, borderValue=0).astype(np.int32)
    brightness = 236.0 - 0.85 * values
    brightness[labels == 0] = 236.0
    if remove is not None:
        brightness[remove] = 236.0
        labels = np.where(remove, 0, labels)
    rng = np.random.default_rng(seed)
    brightness = np.clip(brightness + rng.normal(0, 2.0, brightness.shape), 0, 255)
    gray = brightness.astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1)), labels
