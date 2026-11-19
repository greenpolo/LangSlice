"""The two emlddmm tools as the model sees them: descriptions and argument stubs."""

from __future__ import annotations

from typing import Any, Literal

from langslice.engines.emlddmm.settings import PUBLISHED_SORTING
from langslice.plugin_api import (
    ToolsOn,
)


def stack_description(on: ToolsOn) -> str:
    local_mismatch = on.say_any(
        ("fireants_syn", "landmark_triangulation", "trace_borders"),
        "; for a local mismatch on one section (a ventricle, a displaced piece) "
        "use a per-section warp",
    )
    landmark_kept = on.say("set_landmarks", ", so a landmark move is kept")
    damage_first = on.say("mark_damage", "; mark them with mark_damage first")
    mask_from_sorting = on.say(
        "make_mask", ", and make_mask can take it as a mask for another fit (source fit)"
    )
    zoom_in = on.say("zoom", ", zoom")
    opened_by = on.say("read_file", ", which read_file opens")
    made_images = on.say("make_image", ", or an image made with make_image")
    mask_made = on.say("make_mask", " (made with make_mask)")
    return f"""\
Fit the whole stack to the atlas at once with emlddmm, on its current plane.

emlddmm (Tward et al. 2025, Nature Communications) maps whole-brain datasets
of any modality into the Allen CCF. It models every section as an image of one
deformed 3-D atlas, with an EM sorting of missing tissue and artifacts and a
contrast fit in small blocks.

Use this when you have most of a brain's series, closely and evenly
spaced, and want placements that agree across the stack and fit the
brain's overall shape. It bends the brain as a whole, never one section
alone{local_mismatch}.

emlddmm explains every placed section as a picture of the same 3-D atlas:
one smooth 3-D warp of the atlas, one shared scale and shear for the stack,
and for each section its own in-plane turn and shift (its mounting). It
fits all of them together against the sections' images (one section
image, by name), with a contrast fit in small blocks and an EM sorting of every
pixel into matching tissue, background and artifact, so holes, debris and
empty slide are set aside. Neighbouring sections constrain each other: a
section with little anatomy is bent and shifted the way the stack is. The
cutting angles and positions stay as they are: put the sections in order
and within about one interval of their place first. emlddmm has no
centring step and cannot recover from a start 1 to 3 mm off the tissue,
so have the atlas outline over each section's tissue, placed and
roughly aligned, before calling; each section is
refined from its current registration, its deformation included{landmark_kept}. The section's \
marked damage regions are
left out automatically{damage_first}.

emlddmm takes a section mask as soft weights: the mask multiplies each
pixel's weight in the fit, so a pixel at 0.3 counts at 0.3. Besides the
mask, emlddmm estimates its own sorting during the fit; the final sorting
of each section it writes a deformation for is saved with that
deformation{mask_from_sorting}. It keeps its own brightness scaling: the section image as it
is, 0 to 1.

Writes, per section: with fit "transform" or "both" its transform (turn,
shift, and the stack's scale and shear) and with fit "warp" or "both" its
deformation, the section's cut of the 3-D warp. A section that holds a deformation or a
frozen transform keeps its transform and gets the rigid change folded
into its deformation. A section whose result cannot be written still
shapes the stack. The whole landing is one undo step.

The sections are put on a grid of slots slot_um apart along the slicing
axis, each moved to its nearest slot for the fit (at most half a slot),
and the memory and time grow as the grid gets finer. The call is refused
when the stack would not fit the memory, and says which spacing would.

Starts one piece of background work over the stack and returns at once
with its id. Every section in the stack refuses writes until it lands;
you can look{zoom_in} and work on sections you left out. When it lands, the
next tool reply starts with its notice: how many sections were written,
those fitted for the stack and not written with the reason, the sections
whose warp has plausibility flags with their codes, and the overlays of
up to four sections (flagged ones first, then those that moved most) with
how far each moved. Judge the rest with look. emlddmm's own energy record,
the images it saw and the per-section results (movement, warp
displacement, fold fraction, flags, tissue set aside) are files in the
job{opened_by}.

Args:
    sections: Filenames; empty means every section with a position and a
        transform. At least two.
    slot_um: The slot spacing in micrometres; you set it, there is no
        default. Get the order right and each section within about one
        interval of its place first. Then set slot_um to the section
        thickness, or to about half the interval between neighbouring
        sections. Finer costs memory that many machines lack; coarser
        moves sections further from where you placed them and limits what
        the fit can resolve along the slicing axis. Two sections may not
        share a slot.
    section_image: The section image to fit, by name: "preprocessed", a raw channel{made_images}. \
Read as grey, as it
        is, 0 to 1. Default "preprocessed".
    atlas_image: One of the atlas's images (the job statement lists
        them), or "nissl", the atlas's Nissl (its template when it ships
        none). Default "nissl"; the reply names the image used.
    mask: "none", or the name of a mask that every section in the stack
        has{mask_made}; its weights multiply the
        fit's weight on that section. Default "none".
    kernel_um: The width of the kernel that smooths the 3-D warp's
        velocity, in micrometres, across sections too (emlddmm's a; not a
        Gaussian sigma on the displacement, as smoothness_um is
        elsewhere): tissue closer than this bends together. Default 500.
    fit: "transform" refits each section's in-plane turn and shift and
        the stack's scale and shear and writes them; "warp" keeps every
        transform and fits the 3-D warp alone; "both" fits both. Default
        "both".
    warp_freedom: How freely the warp may bend; larger bends more. Start
        large and reduce if the warps look unrealistic. Default 10000.
    iterations: Steps at the 200 um scale; the 100 and 50 um scales take
        half and a quarter as many. Default 100.
    sorting: Any of "match_sd", "background_sd", "artifact_sd",
        "background_mean", "artifact_mean" (brightness from 0, black, to 1,
        white); keys left out take emlddmm's values. emlddmm's sorting is
        its own EM with an artifact class: it takes no "method" or
        "artifact".
    view: True sends the overlays with the notice; false sends none.

Returns:
    The work id, the number of sections in the stack, those left out with
    why, the sections that will not get a transform and what they get, the
    furthest a section moves to its slot, the device, and per section what
    the fit uses (inputs: the section image, the atlas image as asked and as
    used, the mask, the damage, the brightness scaling); or why it was not
    started. The notice that follows brings, per section, what emlddmm
    estimated (its sorting and the share of the tissue it left out)."""


def emlddmm_stack(
    sections: list[str],
    slot_um: float,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    kernel_um: float = 500.0,
    fit: Literal["transform", "warp", "both"] = "both",
    warp_freedom: float = 10000.0,
    iterations: int = 100,
    sorting: dict[str, Any] = PUBLISHED_SORTING,
    view: bool = True,
) -> dict[str, Any]: ...


def plane_description(on: ToolsOn) -> str:
    write_with = on.say("position_sections", " with position_sections")
    flip_with = on.say("interactive_transform", " with interactive_transform")
    opened_by = on.say("read_file", ", which read_file opens")
    like_stack = on.say("emlddmm_stack", ", as for emlddmm_stack")
    return f"""\
Estimate the stack's cutting angles and spacing with emlddmm. Writes nothing.

Fits emlddmm's 3-D affine to the whole placed stack at once, with the
plane free and no warp: every section votes on one pitch and yaw, and
the spacing may stretch and shift as a whole (a section cannot move on
its own). Starts from the current angles and positions, so place and
orient every section first, with the atlas outline over its tissue (emlddmm
cannot recover from a start 1 to 3 mm off); one call is one descent and lands short
from far away, so write the angles{write_with} and run it again until two passes agree within a
degree or two. emlddmm cannot flip a section: a section mounted mirrored
and not yet flipped pulls the yaw toward zero and stands out from the
others in its residual turn or shift here; flip it{flip_with} and run again before writing. Keep the
positions you placed unless the spacing change looks right in the
pictures. emlddmm's energy record and the images it saw are files in the
job{opened_by}.

Args:
    sections: Filenames; empty means every section with a position. At
        least two.
    slot_um: The slot spacing in micrometres{like_stack}: you
        set it, there is no default. With the order right and each section
        within about one interval of its place, set it to the section
        thickness, or to about half the interval between neighbouring
        sections.
    section_image: The section image to fit{like_stack}.
    atlas_image: One of the atlas's images or "nissl", as for the fit.
    iterations: Steps at the coarsest scale{like_stack}. Default
        100.
    view: True returns the pictures described below; false returns none.

Returns:
    pitch_deg, yaw_deg and their change from the stack's angles; the
    spacing scale and shift_mm; per section what the fit used (inputs), its
    proposed position_mm, the
    change, and the residual turn (deg) and shift (mm) it needed at the
    found plane beyond what the stack as a whole needed; and the overlays of up to four sections \
drawn at the found plane, not written."""


def emlddmm_plane(
    sections: list[str],
    slot_um: float,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    iterations: int = 100,
    view: bool = True,
) -> dict[str, Any]: ...
