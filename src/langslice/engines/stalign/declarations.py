"""The two STalign tools as the model sees them: descriptions and argument stubs."""

from __future__ import annotations

from typing import Any, Literal

from langslice.engines.stalign.settings import MASK_ENGINES, PUBLISHED_SORTING
from langslice.plugin_api import (
    ToolsOn,
)


def lddmm_description(on: ToolsOn) -> str:
    rough_start = on.say(
        "interactive_transform",
        ": set the orientation and a rough transform with interactive_transform first",
    )
    landmark_kept = on.say(
        "set_landmarks", ", a landmark warp included, so the landmark move is kept"
    )
    damage_first = on.say("mark_damage", "; mark them with mark_damage first")
    mask_from_sorting = on.say(
        "make_mask", ", and make_mask can take it as a mask for another fit (source fit)"
    )
    mask_engines = on.say_any(
        MASK_ENGINES,
        "; the engines that take one are " + on.listed([(name, name) for name in MASK_ENGINES]),
    )
    refine_regions = on.say(
        "fireants_syn",
        "; refine single regions afterwards with fireants_syn, which builds on this fit",
    )
    opened_by = on.say("read_file", ", which read_file opens")
    made_images = on.say("make_image", ", or an image made with make_image")
    return f"""\
Fit the atlas onto sections with STalign, on their current registration.

STalign (Clifton et al. 2023, Nature Communications) is LDDMM registration
(large deformation diffeomorphic metric mapping) built to align single-cell
spatial transcriptomics, as rasterised cell images or histology, to each other
and to the Allen CCF. Its distinctive parts are an estimated contrast map, so
the two images need not share brightness, and its own per-pixel sorting into
matching tissue, background and artifact. It is a gradient descent that finds
the nearest fit, so it needs a close start.

STalign bends the atlas image onto each section's image by gradient
descent on their brightness mismatch, with a linear contrast fit between
the two, a kernel width (kernel_um: tissue closer than that bends
together) and a displacement scale (displacement_um: how far a point may
be carried before the penalty grows). It starts from the section's
current registration (its linear placement, and its deformation when it
has one) and refines it; it does not search from scratch{rough_start}. With fit "warp" the \
transform stays and
only the bend is fitted, on top of the deformation the section holds{landmark_kept}. With fit "both"
it also refits the section's in-plane transform and writes it; refused on a
section that holds a deformation, since a new transform clears it. The
section's marked damage regions are left out automatically{damage_first}.

STalign takes no section mask: it sorts the section's pixels itself. Before
each comparison every pixel is weighted between matching (compared),
background (slide, ignored) and artifact (debris, ignored), from the
mismatch and from the background and artifact means (sorting, below). The
final sorting is saved with the deformation{mask_from_sorting}. A named mask is
refused{mask_engines}. STalign fits the whole
section: it takes no regions{refine_regions}. It keeps its own brightness
scaling: slide 0, typical tissue 1.

Starts one piece of background work per section and returns at once with
their ids, so you can carry on with other sections. A write to a section
whose fit still runs is refused until it lands. When a section's fit
lands, the next tool reply starts with its notice: what was fitted, the
displacement, fold fraction and flags, what STalign used and estimated,
and the file in the job folder that holds STalign's own mismatch record
for the fit{opened_by}; then its pictures. Each
landing is its own undo step. status lists the work still running; submit
waits for it. Judge the overlay against the internal anatomy.

Args:
    sections: One to twenty filenames (the extension may be left off).
    section_image: The section image to fit, by name: "preprocessed", a raw channel{made_images}. \
STalign reads it as grey.
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names), or "nissl", the atlas's Nissl (its template when
        it ships none). Default "nissl"; the reply names the image used.
    mask: "none". STalign takes no section mask (see above).
    kernel_um: The width of the kernel that smooths the bend's velocity,
        in micrometres (STalign's a; not a Gaussian sigma on the
        displacement, as smoothness_um is elsewhere): larger bends broader
        patches as one. Default 500.
    fit: "warp" bends on top of the section's registration and keeps its
        transform; "both" also refits the in-plane transform (turn,
        stretch, shift) and writes it. Default "warp".
    displacement_um: The displacement scale in micrometres; larger allows
        a bigger bend. Default 94.
    sorting: {{"method": "classic" | "em" | "none"}} and any of "match_sd",
        "background_sd", "artifact_sd" (spreads), "background_mean",
        "artifact_mean" (left out: estimated) and "artifact" (true adds the
        artifact bin to "em"). Brightness units: slide 0, typical tissue 1.
        "classic" is STalign's published sorting with fixed spreads; "em"
        measures the matching spread itself and takes the background mean
        from the slide at the image border, for sections with missing
        tissue; "none" compares every pixel. Default {{"method": "classic"}}
        with STalign's own spreads.
    iterations: Gradient-descent steps. Default 2000.
    view: True sends the pictures described below with each notice; false
        sends none.

Returns:
    Per section its work id and what the fit uses (inputs: the section image,
    the atlas image as asked and as used, the mask, the damage, the
    brightness scaling), or why it was not started. Each notice brings,
    per section, the numbers above, what STalign ran at (engine_settings:
    its target and atlas grids in micrometres and its own settings), what
    STalign estimated (its sorting,
    saved with the deformation, and the share of the tissue it left out)
    and the section under its new registration with the atlas borders on
    it; when the sorting set some tissue aside, also the final sorting
    picture (set-aside patches outlined; white compared, black ignored),
    and when it set nothing aside the notice says so and sends no picture."""


def stalign_lddmm(
    sections: list[str],
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    kernel_um: float = 500.0,
    fit: Literal["warp", "both"] = "warp",
    displacement_um: float = 94.0,
    sorting: dict[str, Any] = PUBLISHED_SORTING,
    iterations: int = 2000,
    view: bool = True,
) -> dict[str, Any]: ...


def plane_search_description(on: ToolsOn) -> str:
    write_with = on.say("position_sections", " with position_sections")
    flip_with = on.say("interactive_transform", " with interactive_transform")
    return f"""\
Estimate the cutting angles with STalign's plane search. Writes nothing.

For each section, STalign tilts the atlas plane the section sits on by -6,
0 and 6 degrees about each in-plane axis and shifts it by -300, 0 and 300
um along the slicing axis (27 trial planes), fits each roughly to the
section's image at low resolution, refines the best, and
reports where that plane lies: its pitch and yaw in this stack's
convention, and the position it implies. It starts from the section's
current position, the stack's cutting angles and its transform, so place
and orient the section first. Marked damage is left out automatically.
One search reaches about 6 degrees from the start; from further away it
lands short, so write the angles and search again from them until two
passes agree within a degree or two.

The stack has one plane. Search sections with plenty of anatomy spread
along the stack, take the middle value of the pitches and of the yaw
magnitudes, leave out a section flagged at_search_edge or whose picture
at the found plane looks wrong, and write the pair{write_with}. Keep the positions you placed \
unless a
section's picture shows its position change is right. A section mounted
mirrored reverses the sign of its yaw and keeps its pitch, and the
hemispheres often cannot be told apart by any other sign, so many
sections of a stack, up to half, may be mirrored: the reply sorts the
sections by the sign of their yaw. Make every section agree on one sign
(mirror the smaller group{flip_with} and search them again) before writing the angles.

Args:
    sections: One to eight filenames (the extension may be left off).
    section_image: The section image to search with, as for the fits.
    atlas_image: One of the atlas's images, or "nissl", as for the fits.
    view: True returns the pictures described below; false returns none.

Returns:
    Per section pitch_deg, yaw_deg and position_mm of the found plane,
    change (the differences from the current pitch, yaw and position)
    and at_search_edge; the sections grouped by the sign of their yaw,
    with the middle pitch and yaw magnitude of the call; and a picture of
    each section with the atlas borders drawn at the found plane, not
    written."""


def stalign_plane_search(
    sections: list[str],
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    view: bool = True,
) -> dict[str, Any]: ...
