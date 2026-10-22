"""Contextual template / preprocessed-section comparisons for saved pairs."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from PIL import Image

from langslice.core import look
from langslice.core.landmarks import glyph, section_picture_point, transform
from langslice.core.layers import note
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace


def pair_picture(
    ws: Workspace, state: StackState, record: SliceState, pair: dict[str, Any], *, store: Any = None
) -> look.LookPicture:
    """One crop per endpoint, retaining at least half the full anatomical extent."""
    pictures = []
    for role, mode in (("atlas", "atlas"), ("section", "section")):
        request = look.LookRequest(
            mode=mode,
            sections=(record.id,),
            images=("preprocessed",) if role == "section" else (),
            atlas_layers=("template", "borders") if role == "atlas" else ("template",),
            long_edge=512,
        )
        whole = look.look(ws, state, request, store=store)[0]
        panel = whole.recipe["endpoint_panels"][0]
        point = (
            transform(np.linalg.inv(panel["pixel_to_native"]), pair["atlas_native_xy"])
            if role == "atlas"
            else section_picture_point(panel, pair["section_fraction_xy"])
        )
        width, height = whole.recipe["shown"]
        centre = point / np.array([width, height])
        # Keep a 60% window even at an image edge: shift it instead of
        # trimming away the anatomy on the far side of the landmark.
        lower = np.clip(centre - 0.3, 0, 0.4)
        upper = lower + 0.6
        pictures.append(
            look.look(
                ws,
                state,
                look.LookRequest.from_args({**request.args(), "zoom": [*lower, *upper]}),
                store=store,
            )[0]
        )
    for role, picture in zip(("atlas", "section"), pictures, strict=True):
        panel = picture.recipe["endpoint_panels"][0]
        point = (
            transform(np.linalg.inv(panel["pixel_to_native"]), pair["atlas_native_xy"])
            if role == "atlas"
            else section_picture_point(panel, pair["section_fraction_xy"])
        )
        glyph(picture.image, point, f"{pair['id']} selected", role)
    left, right = pictures
    gap = 24
    image = Image.new(
        "RGB",
        (left.image.width + gap + right.image.width, max(left.image.height, right.image.height)),
        "#333333",
    )
    image.paste(left.image, (0, 0))
    offset = left.image.width + gap
    image.paste(right.image, (offset, 0))
    panels = copy.deepcopy(left.recipe["endpoint_panels"])
    for panel in copy.deepcopy(right.recipe["endpoint_panels"]):
        panel["bounds"][0] += offset
        panel["bounds"][2] += offset
        to_local = np.array([[1, 0, -offset], [0, 1, 0], [0, 0, 1]])
        panel["pixel_to_file_fraction"] = (
            np.asarray(panel["pixel_to_file_fraction"]) @ to_local
        ).tolist()
        panels.append(panel)
    text = f"{record.id} {pair['id']}: atlas template (diamond), preprocessed section (circle)"
    recipe = {
        "renderer": "landmark_pair",
        "endpoint_panels": panels,
        "registration_context": left.recipe["registration_context"],
        "shown": list(image.size),
        "state": left.recipe["state"],
    }
    from langslice.core.captions import caption

    image = caption(image, text)
    note(image, sections=(record.id,), mode="landmarks", recipe=recipe, caption=text)
    return look.LookPicture(
        image=image,
        caption=text,
        recipe=recipe,
        sections=(record.id,),
        mode="landmarks",
        um_per_px=0,
    )
