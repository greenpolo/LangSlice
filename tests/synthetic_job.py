"""Three synthetic sections with known positions: the job the library tests open."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section

SECTION_SHRINK = 0.8
PIXEL_SIZE_UM = 25.0 / SECTION_SHRINK
POSITIONS = {"s0.png": 0.1, "s1.png": 0.15, "s2.png": 0.2}


def write_sections(folder: Path) -> None:
    """Three grey sections of the synthetic atlas under a smooth known field."""
    atlas = SyntheticAtlas()
    folder.mkdir(parents=True)
    for seed, name in enumerate(POSITIONS):
        drawn, _ = render_section(atlas, SMOOTH_FIELD(), seed=seed)
        plane = cv2.GaussianBlur(np.asarray(drawn)[..., 0], (0, 0), 1.2)
        size = (
            round(plane.shape[1] * SECTION_SHRINK),
            round(plane.shape[0] * SECTION_SHRINK),
        )
        small = cv2.resize(plane, size, interpolation=cv2.INTER_AREA)
        Image.fromarray(np.stack([small] * 3, axis=-1)).save(folder / name)
