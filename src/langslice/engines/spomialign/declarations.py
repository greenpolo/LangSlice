"""The ``spomialign`` tool as the model sees it: its description and its argument stub."""

from __future__ import annotations

from typing import Any, Literal

from langslice.plugin_api import (
    ToolsOn,
)


def spomialign_description(on: ToolsOn) -> str:
    steps = on.listed(
        [
            ("position_sections", "place the section with position_sections"),
            (
                "interactive_transform",
                "set its flip, quarter turn and rough transform with interactive_transform",
            ),
        ]
    )
    rough_start = f": {steps} first" if steps else ""
    damage_first = on.say(
        "mark_damage",
        "; where tissue is missing or lies folded over other anatomy, mark those regions "
        "with mark_damage first (tissue that is intact but shifted is not damage)",
    )
    landmark_kept = on.say(
        "set_landmarks", ", a landmark warp included, so the landmark move is kept"
    )
    made_images = on.say("make_image", ", or an image made with make_image")
    mask_made = on.say("make_mask", " (made with make_mask)")
    opened_by = on.say("read_file", ", which read_file opens")
    return f"""\
Register sections to the atlas with SPOmiAlign, on their current registration.

SPOmiAlign (Wang et al. 2026, Briefings in Bioinformatics) was built to align
spatial-omics sections (transcriptomics, proteomics, H&E) to each other and to
the Allen Nissl plates. Instead of comparing brightness it matches points with
RoMa (Edstedt et al. 2024), a dense feature matcher pretrained on DINOv2
features, so only the tissue's shape and structure need agree, not its stain.

SPOmiAlign matches distinctive points between a section image (by
name) and an atlas image with a pretrained dense matcher (RoMa), weights
the matches toward tissue edges, keeps the most confident (matches), and
fits the chosen transform to the pairs; it compares no intensities pixel by
pixel and runs no iterations; a call takes under a minute. It starts from
the section's current placement and does not search from scratch{rough_start}. The section's \
marked damage regions are left
out automatically{damage_first}.

fit chooses what is fitted:
- "transform": the section's transform: rotation, shift, scale and shear
  (an affine), or with rigid rotation and shift alone, the section keeping
  its size. Written as the section's transform.
- "warp": a smooth local deformation on top of the registration the section
  holds (its transform, and its deformation when it has one{landmark_kept}); the transform
  stays. Written as the section's deformation.
- "both": an affine, then a warp on top of it (SPOmiAlign's own default).
  Writes both.
A new transform clears the section's deformation, so "transform" and
"both" are refused on a section that holds one: fit "warp" on it instead.

The warp is SPOmiAlign's smoothed match field: each pair's displacement,
filled in between the pairs and smoothed by a Gaussian of smoothness_um.
Where the section and the atlas image share little tissue (a damaged or
partial section), the affine follows wrong matches, because it weighs
every pair equally and rejects none; the sign is an atlas shrunk off the
tissue in the returned overlay. For such a section, fit "transform" with
rigid, check the returned overlay, then fit "warp". The warp bends whole
regions and does not refine individual borders.

SPOmiAlign reads one image pair per section, nothing else, and has no
weighting: it estimates no mask of its own, and a section mask can only
drop matches. With a mask named, every match whose end on the section lies
where the mask is below one half is dropped and the next most confident
take their place; the mask is approximated by dropping matches, not
applied to the pixels. The atlas side's marked damage is dropped the same
way. The section is read as 8-bit RGB with the atlas image drawn as a
brightfield Nissl, dark on white.

Each call is one undoable write.

Args:
    sections: One to twenty filenames (the extension may be left off).
    fit: "transform", "warp" or "both".
    rigid: True fits the transform as a rotation and a shift alone (fit
        "transform" only). Default false.
    section_image: The section image to match, by name: "preprocessed", a raw \
channel{made_images}. Default "preprocessed".
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names), or "nissl", the atlas's Nissl (its template when
        it ships none). Default "nissl"; the reply names the image used.
    mask: "none", or the name of a mask of each section{mask_made}; matches ending where it is \
below one half
        are dropped. Default "none".
    smoothness_um: The width (sigma) of the Gaussian that smooths the
        warp's displacement field, in micrometres of the atlas; larger
        bends broader regions as one. Default 150.
    matches: How many of the most confident matched pairs the fit uses.
        Default 5000 (SPOmiAlign's).
    view: True returns the picture described below; false returns none.

Returns:
    Per section what was fitted and on what, what the fit used (inputs:
    the section image, the atlas image as asked and as used, the mask, the
    damage, the brightness scaling), what SPOmiAlign ran at
    (engine_settings: the matcher, its working grid and micrometres per
    pixel, the smoothing in its pixels, the pairs), the transform written
    (rotation_deg, scale_x, scale_y, shear, translate_x_mm,
    translate_y_mm about the canvas centre) when one was, for a warp its
    displacement (median and max, mm), fold fraction and plausibility
    flags, the pairs used and dropped, and the folder in the job holding
    the call's files (the two images SPOmiAlign saw, its settings and the
    matched pairs as a table and a picture{opened_by}); and a picture of each section under its \
new registration
    with the atlas borders on it. SPOmiAlign estimates no weighting, so
    estimated is null."""


def spomialign(
    sections: list[str],
    fit: Literal["transform", "warp", "both"],
    rigid: bool = False,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    smoothness_um: float = 150.0,
    matches: int = 5000,
    view: bool = True,
) -> dict[str, Any]: ...
