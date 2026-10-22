"""Every verb's declaration: the one home of what each door offers.

Each verb of
:data:`langslice.ops.registry.VERBS` is declared here ONCE, as a stub
function below: its arguments (names, types, defaults) and its description.
Every door is generated from these declarations:

- the agent tools and the MCP tools: ``langslice.doors.tools.toolbox.build_tools``
  puts each tool body behind :func:`declare`, so ADK's function declaration
  and FastMCP's input schema are read off these signatures and docstrings;
- the agent CLI: ``langslice-job ops`` (:func:`summary`), ``langslice-job schema
  VERB`` (:func:`arguments_schema`), ``langslice-job FOLDER VERB``;
- the job folder's reference card and the library's job methods.

The description is the stub's docstring, which a model reads verbatim (ADK
and FastMCP send a function's ``__doc__`` as it is), and it is the ONLY
description of the tool: the job statement names the tools and describes
none. Every continuation line a model reads is indented eight spaces;
:func:`model_doc` adds the four this module level lacks (the text snapshot pins
the text).

A run varies a declaration in the ways a :class:`Variant` names: ``look``'s and
``landmark_triangulation``'s ``resolution`` exist only where the agent chooses
the picture size (image resolution "auto"; the CLI always); the change tools'
``view`` is not offered
where the host forces their pictures on (``JobSpec.force_view``);
``position_sections`` takes ``sections`` only with the Positioning task;
``trace_borders`` carries the run's base image
prompt; each door words where its answers come from (``_DOOR_DOCS``); and
what the job disables (``JobSpec.disabled``) is gone: an option from the
argument's enum (``look``'s ``mode``), the arguments only it served
(:func:`dropped_arguments`), and every passage that needs it. A passage
that names a tool or an option is written with what it needs
(``[[?look:positioning ...]]``, :mod:`langslice.core.needs`) and rendered
for the run. An argument a run drops leaves the description with its entry.

Door-only parameters (ADK's ``tool_context``) are the tool body's, never
declared. Nothing here imports ``google.*``, ``litellm`` or ``openai``.
"""

from __future__ import annotations

import functools
import inspect
import re
import typing
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Literal

from langslice.core import section_images as images
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.deformable.settings import DEFAULT_NEIGHBOURHOOD_UM
from langslice.core.needs import render
from langslice.core.toolset import OPTION_SEPARATOR, TOOLS, is_off, options_on
from langslice.doors.tools.arguments import (
    CuttingAngles,
    LandmarkPair,
    PicturePoint,
    SectionPosition,
    SectionTransform,
    SectionTransformNoFlip,
    SectionTransformNoOrientation,
    SectionTransformNoRotate,
)

# Defaults of list and object arguments are read, never mutated (ADK wants a value).
# ruff: noqa: B006

#: The choices a string argument takes, offered as an enum in every door's schema.
LookMode = Literal["section", "atlas", "overlay", "positioning"]
#: Atlas layers and fit images are the atlas's own (its "template" and any
#: additional references BrainGlobe ships), so they differ by atlas: plain
#: strings, listed in the job statement and checked by the tools.
AtlasLayer = str
Warp = Literal["applied", "none"]
FitAtlas = str
MaskSource = Literal["em", "fit"]
FitMetric = Literal["mutual_information", "local_correlation"]
FitDetail = Literal["coarse", "standard", "fine"]

# --- looking ---------------------------------------------------------------------


def look(
    mode: LookMode,
    sections: list[str] = [],
    positions_mm: list[float] = [],
    images: list[str] = [],
    atlas_layers: list[AtlasLayer] = [],
    atlas_opacity: float = 0.5,
    warp: Warp = "applied",
    resolution: int = 0,
    layers: list[str] = [],
    depth: int = OVERLAY_DEPTH,
    masks: list[str] = [],
) -> dict[str, Any]:
    """Draw pictures of sections and the atlas. Writes nothing.

    The same request draws the same kind of picture at every stage of the
    run: no default depends on the stack's state. Each picture gets a number
    and a caption stating the positions, cutting angles, scale and section
    image it was drawn with[[?zoom ; zoom takes the number]].

    [[?look:positioning Use positioning mode to compare sections with candidate atlas planes
    along the slicing axis. ]][[?look:overlay Use overlays to assess in-plane alignment and
    deformation against internal anatomy.]]

    Args:
        mode: [[?look:section "section": each section alone, as oriented, the whole
            image. ]][[?look:atlas "atlas": the atlas plane alone at each of positions_mm,
            at the stack's cutting angles, or with one section at that
            section's own plane (positions_mm may be empty). ]]~
            [[?look:overlay "overlay": each section under its
            current registration, with the atlas borders and any other
            atlas_layers on it. ]][[?look:positioning "positioning": the sections and the atlas
            along the slicing axis as ABBA lays them out: the atlas at
            positions_mm above a millimetre ruler, the sections below it in
            order of position, each placed section joined by a line to its
            position. Unplaced sections appear at the right, labelled "no
            position", without a line to the ruler. A long stack is split
            into several pictures, none shrunk.]]
        sections: Filenames (the extension may be left off); empty is every
            section.
        positions_mm: Atlas positions in millimetres along the slicing axis:
            [[?look:atlas the planes "atlas" draws (at least one)~
            [[?look:positioning , and ]]]]~
            [[?look:positioning the atlas pictures
            above the ruler in "positioning" (none when empty)]].
        images: Section images to draw, by name (status lists them): one image
            alone, in gray, or up to 6 raw channels together, each in its own
            colour with automatic contrast. Empty draws `preprocessed`.
        atlas_layers: Any of the atlas's images (the job statement lists
            them: "template" and any other reference images the atlas ships,
            by their names) and "borders". Empty is the mode's default: ~
            [[?look:overlay borders in "overlay"]]~
            [[?look:overlay,look:atlas|look:positioning ,
            ]]~
            [[?look:atlas,look:positioning template in "atlas" and "positioning"||~
            [[?look:atlas template in "atlas"]]~
            [[?look:positioning template in "positioning"]]]].~
            [[?look:section  "section" draws no atlas.]]
        atlas_opacity: 0 to 1: how strongly an atlas image shows under the
            section in "overlay".
        warp: "applied" draws each section with its deformation, "none" with
            its linear placement only.
        resolution: The long edge of each picture in pixels; 0 is the
            default size. A section picture is never drawn larger than its
            source, an atlas plane at most four times larger.
        layers: Names of the layers left on the sections (status lists them),
            or ["all"], drawn as marks and outlines on the [[#look:section=\"section\"||~
            look:overlay=\"overlay\"]] pictures. The reply says which layers
            were drawn. Empty draws none.
        depth: Atlas ontology depth whose region boundaries "borders"
            draws (0 = root, as in BrainGlobe; larger is finer).
        masks: [[?make_mask Names of masks made with make_mask, outlined where
            their weight is 0.5 on the sections drawn, on the [[#look:section=\"section\"||~
            look:overlay=\"overlay\"]] pictures. Empty draws none.]]

    Returns:
        Up to 4 pictures, attached in the order `pictures` lists them, each
        with its number and caption. Pictures past 4 are listed under
        `not_shown` with the call that draws them.
    """
    ...


def zoom(box: list[float] = [], picture: int = 0, region: str = "") -> dict[str, Any]:
    """Zoom into a pixel box or one atlas region of an earlier picture. Writes nothing.

    The new picture is as large as the earlier one. In a picture of a
    section or of a section with the atlas on it, the section is read again
    from its original image file at the file's own resolution, only inside
    the box, and drawn with the same section image and atlas
    borders; the caption says when the file holds fewer pixels than the
    picture. [[?look:positioning A positioning picture's sections are drawn again from their
    working copies (the file reduced to at most 3072 pixels), and the atlas
    from its voxels. ]]The box is drawn as the picture was, even if the stack
    has changed since (the reply then says `stale`). The new picture has its
    own number, so it can be zoomed again. A dashed green grid on it is
    labelled with the pixel numbers of the whole picture the zoom was cut
    from, and those numbers are the coordinates of a further zoom's box[[?set_landmarks  and
    of a set_landmarks endpoint, on the zoom or on the whole picture alike]].

    [[?look:overlay After a linear or nonlinear fit, use region on its overlay to inspect
    a particular structure closely, with only that region's border shown.
    This can reveal internal mismatches hidden at whole-section scale and
    help you decide whether to adjust the linear alignment, refine that
    region nonlinearly, or revisit the section's position.

    ]]Earlier pictures remain available by number. [[?look:overlay Reopen or zoom a saved
    overlay to compare an intermediate registration with the current result;
    its saved placement is preserved. ]]Use undo when the latest change makes
    the alignment worse.

    Args:
        box: [x0, y0, x1, y1] in pixels of that picture as it was shown,
            from its top-left corner (on a zoom, the numbers its grid prints).~
            [[?look:overlay  Give either box or region, not both.]]
        picture: The picture's number; 0 is the newest picture that is not
            itself a zoom.
        region: One atlas acronym or numeric id, descendants included;
            "CTX:left" or "CTX:right" selects one displayed side. Overlay
            pictures only. Crops to the entire region with a small margin
            and draws only its border, preserving the saved registration
            and section image. An absent region is reported. Leave box
            empty when using region.

    Returns:
        One picture with its number, caption, `covers` [x0, y0, x1, y1] (the
        whole picture's pixels it shows) and `content_box` [x0, y0, x1,
        y1], where the drawing lies in the image (a narrow box is padded with
        black beside it, the caption below), and `redrawn`: false when
        the picture could not be drawn again, so the box was cut from the
        saved picture and enlarged, with no more detail than it had.
    """
    ...


def make_image(
    name: str,
    sections: list[str] = [],
    weights: dict[str, float] = {},
    clahe_clip: float = images.DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = images.DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    contrast: list[float] = [],
    gamma: float = 0.0,
    invert: bool = False,
    delete: bool = False,
    view: bool = True,
) -> dict[str, Any]:
    """Make, replace or delete a named section image, from the section's other images.

    A section image is a grayscale picture of a section that look draws and
    the fits read by name: the raw channels of the file, never edited, and the
    images made here, each saved under the job folder
    (sections/<name>/images/<name>.png; status lists them). A fit reads the
    file as it is, so a look at an image is what the fit sees. "preprocessed"
    was made when the job opened; make it again, or make others under new
    names, as the sections need. The steps run in this order on each source:
    n4, denoise, CLAHE; then the weighted blend, brightened to a mid-gray
    tissue mean; then contrast, gamma, invert. Undoable.

    Args:
        name: The image's name (not a raw channel's); an existing name is
            replaced.
        sections: Filenames (the extension may be left off) of the sections to
            make it for; empty is every section. A later call can remake it for
            some sections alone.
        weights: The images to blend and their weights, by name: {"dapi": 1},
            {"red": 1}, {"dapi": 1, "gfp": 0.3}, or a made image's name. Empty:
            each raw channel weighted by how much of the tissue it lights.
        clahe_clip: Local contrast enhancement (CLAHE) clip limit, 0 (none) to
            40.
        clahe_tiles: CLAHE tiles per side, 1 to 32.
        n4: Correct uneven illumination (N4).
        denoise: Patch-based denoising.
        contrast: [low, high], fractions 0 to 1 of the blended image's range,
            stretched to the full range; empty leaves it.
        gamma: 0.1 to 10; 1 or 0 is linear.
        invert: Bright as dark.
        delete: True removes the image (from the named sections, or
            everywhere); the other arguments are then ignored.
        view: True returns the pictures described below; false returns none.

    Returns:
        The recipe now in force and the section images there are[[?look:section , and with
        view up to 4 of the sections (spread over the stack when none are
        named), each drawn as look's "section" mode draws the image]].
    """
    ...


def grep_atlas(query: str = "", section: str = "",
               points: list[PicturePoint] = []) -> dict[str, Any]:
    """Look regions up in the atlas hierarchy by acronym, name or id. Text only.

    When you recognize a structure in the tissue, find its atlas name here[[?grep_atlas_view ,
    then use grep_atlas_view to check its shape and location at candidate
    positions. Confirm the region visually before using it to guide a fit
    [[?mark_damage or mark damage]]]].

    [[?grep_atlas:points With points, it also says which atlas region is drawn at given points of
    saved pictures: the region (acronym, name, id) and the regions containing it,
    or that the point is outside the atlas. A picture that draws the atlas gives a
    region[[?look:overlay : an overlay of a section[[?zoom  or a zoom of one]], or an
    atlas picture drawn for a section]]; a picture of the section alone does not.]]

    Args:
        query: Text matched case-insensitively against region acronyms and
            names (substring), or an exact acronym or numeric id[[?grep_atlas:points ;
            empty when only points are given]].
        section: Optional filename (the extension may be left off) of a
            section with a position; each row then says whether the region
            (or any descendant) appears in the atlas plane at that section's
            placement.
        points: [[?grep_atlas:points Points to look the region up at: {picture, x, y}, at
            most 40, x and y in pixels from the top-left of the whole picture~
            [[?zoom  (on a zoom, the numbers its grid prints)]].]]

    Returns:
        Rows of acronym, id, name, ancestry (root to parent, as acronyms)
        and descendant count, at most 40, with the number left over[[?grep_atlas:points ; and,
        per point, its region and containing regions]].
    """
    ...


def grep_atlas_view(regions: list[str], positions_mm: list[float]) -> dict[str, Any]:
    """Draw the atlas at given positions with chosen regions' borders highlighted. Writes nothing.

    Only the named regions' borders are drawn, at the stack's cutting
    angles. Look highlights no regions; this tool does.

    [[?grep_atlas After finding a region with grep_atlas, compare its shape and location
    here with the structure you recognize in the tissue, before using it
    to guide a fit[[?mark_damage  or mark damage]].]]

    Args:
        regions: Acronyms, names or ids, descendants included; "CTX:left" or
            "CTX:right" names one side, as the pictures show it.
        positions_mm: Atlas positions in millimetres along the slicing axis;
            a position outside the atlas is clamped into its range.

    Returns:
        Up to 4 pictures with their numbers and captions, and per position
        the named regions that have no pixel in that plane.
    """
    ...


def status() -> dict[str, Any]:
    """The stack as it stands, and what this run lets you change. Writes nothing.

    Returns:
        One row per section in stack order: filename, position, distance
        to the next placed section, flip and quarter turn, transform,
        damaged regions and damage note, deformation steps[[?make_mask , masks
        (name, source, and whether it was made at an earlier registration)]]. Also the
        cutting angles, the interval breaks, the raw channels and the section
        images made (each with its recipe, or a script's note, and the
        sections it differs on or exists on), the background work still
        running, and what this run lets you change.
    """
    ...


def list_files(path: str = ".", pattern: str = "") -> dict[str, Any]:
    """List the files and folders of the job folder. Writes nothing.

    The job folder keeps every picture you were shown, indexed in
    views.jsonl (number, tool, history step, sections, caption), with the
    run notes (state.json), the settings (job.json) and the results.

    Args:
        path: A folder inside the job folder, relative to it; "." is the job
            folder itself.
        pattern: A glob (e.g. "*.json", "views/*"): every file below path
            whose name or path matches. Empty lists path itself.

    Returns:
        The entries, folders first, each with its kind and size, at most 200,
        with the number left out.
    """
    ...


def search_files(query: str, path: str = ".", glob: str = "") -> dict[str, Any]:
    """Search the job folder's text files for lines that match. Writes nothing.

    Args:
        query: A regular expression, or plain text when it is not one; upper
            and lower case alike.
        path: A folder or file inside the job folder, relative to it.
        glob: Only files whose name or path matches this glob.

    Returns:
        Matches as "file:line: text", at most 60, with the number left out.
        Pictures and binary files are not searched.
    """
    ...


def read_file(path: str, offset: int = 0, limit: int = 400) -> dict[str, Any]:
    """Read a text file of the job folder, with line numbers. Writes nothing.

    A picture file is answered with its entry in the picture index (its
    number, tool, sections and caption), never its pixels[[?zoom : zoom shows it
    again by that number]].

    Args:
        path: The file, relative to the job folder.
        offset: Lines to skip from the start.
        limit: Lines to show, at most 2000.

    Returns:
        The lines shown, the file's line count and, when lines remain, the
        offset to continue from.
    """
    ...


# --- changing ---------------------------------------------------------------------


def position_sections(
    sections: list[SectionPosition] = [],
    cutting_angles: CuttingAngles = {},
    view: bool = True,
) -> dict[str, Any]:
    """Set where sections sit along the slicing axis, and the stack's cutting angles.

    Positions are atlas millimetres along the slicing axis; a value outside
    the atlas range is clamped into it and reported. The stack's order
    follows the positions. A changed position or angle clears the
    deformation of each section it moves (the reply lists them). One
    undoable write.

    Cutting angles tilt the atlas plane every section is cut at; every
    section gets the same angles, and every later atlas picture is drawn at
    them. Pitch 0 and yaw 0 is the atlas's flat plane. A tilt moves each
    edge of the plane along the slicing axis: an edge D mm from the
    picture's centre moves about D x tan(angle) mm (10 degrees moves an edge
    4 mm from the centre about 0.7 mm). Which edge moves which way, for
    positive angles, by the plane the stack is cut in (edges as the pictures
    show them):

    - coronal: pitch puts the top edge at a larger position (mm) than the
      bottom edge; yaw puts the right edge at a larger position than the
      left edge.
    - sagittal: pitch puts the left edge at a larger position than the
      right edge; yaw puts the bottom edge at a larger position than the
      top edge.
    - horizontal: pitch puts the right edge at a larger position than the
      left edge; yaw puts the top edge at a larger position than the
      bottom edge.

    Negative angles move the edges the opposite way.

    Args:
        sections: [{"id": "<filename>", "position_mm": <number>}]; the
            extension may be left off.
        cutting_angles: {"pitch_deg": <number>, "yaw_deg": <number>} for the
            whole stack; empty leaves the angles as they are.
        view: True returns the picture described below; false returns none.

    Returns:
        What was written and what was clamped, the stack's order after the
        call, the rows that changed, and [[?look:positioning a positioning picture of the
        written sections: the atlas at their new positions above a
        millimetre ruler, the sections below it, each joined by a line to
        its position.||a picture of each written section with the atlas
        borders of its new position on it.]]
    """
    ...


def interactive_transform(
    sections: list[SectionTransform],
    view: bool = True,
) -> dict[str, Any]:
    """Set sections' [[?interactive_transform:flip|interactive_transform:rotate ~
    orientation and ]]in-plane transform by hand, and show the result.

    For a few sections at a time, in a look-and-adjust loop. Values are
    absolute: a value given replaces the section's current one, and a value
    left out keeps it (a section without a transform starts from the
    identity). [[?interactive_transform:rotate The quarter turn is applied first~
    [[?interactive_transform:flip , then the flip]]. ]]~
    [[?interactive_transform:flip|interactive_transform:rotate A changed
    orientation keeps the other values: the transform is rebuilt on the new
    orientation, and the reply says so. ]]One undoable write; the image files
    are never modified.

    [[?interactive_transform:flip ~
    Whether a section is mirrored can only be decided from a visible notch,
    or from a cutting angle oblique enough that the two hemispheres'
    anatomy differs.

    ]]Args:
        sections: One to four objects: "id" (filename; the extension may be
            left off) and any of [[?interactive_transform:flip "flip" (true mirrors the section
            left-right), ]]~
            [[?interactive_transform:rotate "rotate_quarter" (0, 90, 180 or 270 degrees
            counter-clockwise), ]]"rotation_deg" (degrees counter-clockwise
            about the pivot, the canvas centre),
            "scale_x" and "scale_y" (multipliers about the pivot; 1.0 keeps
            the size), "shear" (a slant applied before the rotation: each
            point moves sideways by shear times its distance below the pivot,
            in units of scale_x), "translate_x_mm" (right) and
            "translate_y_mm" (down). A section appears once per call.
        view: True returns the picture described below; false returns none.

    Returns:
        Per section [[?interactive_transform:flip|interactive_transform:rotate ~
        its orientation, ]]its transform as the values above,
        and whether anything was written; and a picture of each section
        under its new transform with the atlas borders on it.
    """
    ...


def mark_damage(section: str, regions: list[str], note: str = "") -> dict[str, Any]:
    """Mark the atlas regions a section has lost, so that every fit leaves them out.

    Mark the regions whose tissue is missing from the section, or lies folded
    over other anatomy, such as a whole hemisphere, the olfactory bulb or the
    cortex: the fits leave them out, and the anatomy around them fits
    cleanly. Tissue that is intact but shifted near its place is not damage;
    left unmarked, the warp fits it.[[?landmark_triangulation  Tissue shifted further is not
    damage either: move it with landmarks.]] A marked region is never fitted, so
    intact tissue inside a mark goes unfitted too. When most of a large
    region is missing, marking the whole region keeps the anatomy around it
    safe. Small tears, bubbles, stains, low contrast and folds within an
    intact outline are not damage and are not marked. A section is damaged
    exactly when it has marked regions. ~
    [[?elastix_affine|fireants_syn|trace_borders The marked
    regions are left out of ~
    [[#elastix_affine=elastix_affine||fireants_syn=fireants_syn||~
    trace_borders=the image model's
    trace]] automatically, and they||The marked regions]] move with the registration because they
    are named by atlas region. One undoable write.

    Args:
        section: Filename (the extension may be left off).
        regions: Acronyms, names or ids, descendants included; "CTX:left" or
            "CTX:right" names one side, as the pictures show the section.
            They replace the section's marked regions; empty clears them.
        note: What is wrong with the tissue. A note the user gave stays
            first.

    Returns:
        The section's marked regions and note, and a picture of the marked
        regions shaded on the section under its current registration beside
        the same regions on the atlas, to check both at once.
    """
    ...


# --- fitting ----------------------------------------------------------------------


def make_mask(
    name: str,
    sections: list[str] = [],
    source: MaskSource = "em",
    section_image: str = "preprocessed",
    atlas_image: FitAtlas = "nissl",
    sorting: dict[str, float] = {},
    delete: bool = False,
    view: bool = True,
) -> dict[str, Any]:
    """Make, replace or delete a named mask of sections: which tissue a fit compares.

    A mask holds a weight from 0 to 1 for each pixel of a section; a fit
    that takes it by name ignores the pixels under 0.5. It sorts the
    section's pixels into tissue, artifact and background by their brightness
    in a section image and by where the atlas puts tissue, as the fits' own
    estimate does, and it is the only way a fit leaves out a hole, a tear or a
    stain. Each section gets its own mask, made at its own current
    registration. The mask sits on the section's own pixels, so it stays usable after
    fits and moves; status says when it was made at an earlier registration.
    Turning or flipping the section makes a fit refuse it until it is made
    again. The marked damage regions are not part of a mask; every
    fit leaves them out of the atlas on its own. Undoable.

    Args:
        name: The mask's name; an existing name is replaced.
        sections: Filenames (the extension may be left off) of the sections to
            make it for; empty is every section.
        source: "em": sort each section's pixels now, at its current
            registration, against atlas_image. "fit": the sorting the
            section's applied deformation saved with it, copied as it was
            (NO_DEFORMATION or NO_SORTING when it has none).
        section_image: The section image to sort ("preprocessed", a raw channel~
            [[?make_image  or an image made with make_image]]); "em" only.
        atlas_image: The atlas image the sorting compares with ("nissl" is the
            atlas's Nissl image where it has one, else its template); "em"
            only.
        sorting: Starting points of the sort, each between 0 and 1 of the
            section image's brightness or a spread in the same units: match_sd,
            background_sd, artifact_sd, background_mean, artifact_mean. Empty
            uses the defaults; "em" only.
        delete: True removes the mask (from the named sections, or
            everywhere); the other arguments except name are then ignored.
        view: True returns the pictures described below; false returns none.

    Returns:
        Per section the mask's source, section image, atlas image and
        starting points, and the fraction of the section's tissue it leaves
        out, or why the section has no mask. With view, up to 4 of the
        sections (spread over the stack when none are named), each drawn as
        look's "section" mode draws the image it sorted, with the mask's 0.5 outline on it,
        and a caption saying the outlines trace the mask's edge, that the
        ringed patches are what the mask leaves out of a fit, and the mask's
        name and source.
    """
    ...


def elastix_affine(
    sections: list[str],
    restrict_to: list[str] = [],
    section_image: str = "preprocessed",
    atlas_image: FitAtlas = "nissl",
    mask: str = "none",
    edges: bool = True,
    view: bool = True,
) -> dict[str, Any]:
    """Fit sections' in-plane affine transforms automatically with elastix.

    For a few sections at a time. Elastix refines each section's current
    placement (from the identity when it has no transform) by matching a
    section image against an atlas image, inner anatomy included. It
    adjusts from where the
    section is and does not search from scratch: it moves a section by
    small amounts and does not turn it over. The section's marked damage
    regions are left out of the atlas automatically. A section that holds a
    deformation is refused with HAS_DEFORMATION: undo the deformation
    first[[?fireants_syn , or fit a warp on it]]. Each fit is written as the
    section's transform, one undoable write for the call.

    Use restrict_to when a few clearly recognizable, surviving regions
    provide better registration evidence than the whole section. Those
    regions guide a whole-section affine transform. Review the wider
    overlay afterward as well.

    A region too small for elastix is skipped with REGION_TOO_SMALL; use
    a larger region[[?interactive_transform  or interactive_transform]]. No fit is written for it.

    Both images are scaled from their 1st to 99th percentile inside the
    atlas outline to 0-1. The reply names the section image, atlas image and mask
    each fit used.

    Args:
        sections: One to four filenames (the extension may be left off).
        restrict_to: Regions to fit by (acronyms, names or ids, descendants
            included; "CTX:left" or "CTX:right" for one side): only the
            atlas within 300 um of them, against the tissue the fit lays
            there. Only the ANTs and Elastix fits take it. Empty fits by
            every region.
        section_image: The section image to match, by name: "preprocessed", a raw channel~
            [[?make_image , or an image made with make_image]].
        atlas_image: One of the atlas's images, by name (the job statement lists
            them and says what each is), or "nissl", the atlas's Nissl image
            where it has one, else its template.
        mask: "none" compares every pixel of the section[[?make_mask , or the
            name of a mask of that section made with make_mask: pixels with
            a weight under 0.5 are left out]].
        edges: True also matches the edges (the gradient magnitude) of both
            images, as a second term of equal weight.
        view: True returns the picture described below; false returns none.

    Returns:
        Per section the transform (rotation_deg, scale_x,
        scale_y, shear, translate_x_mm, translate_y_mm about the canvas
        centre); the `inputs` it used (section image, atlas image as
        resolved, mask, damage regions left out, intensity scaling); and a
        picture of each fitted section
        under its new transform with the atlas borders on it. With
        restrict_to, those regions' borders are drawn thick and the others
        faint, and the picture is zoomed to them when they cover a small
        part of the section.
    """
    ...


def fireants_syn(
    sections: list[str],
    restrict_to: list[str] = [],
    section_image: str = "preprocessed",
    atlas_image: FitAtlas = "nissl",
    mask: str = "none",
    smoothness_um: float = 80.0,
    metric: FitMetric = "mutual_information",
    edges: bool = True,
    detail: FitDetail = "standard",
    view: bool = True,
) -> dict[str, Any]:
    """Fit the atlas onto sections with FireANTs SyN, on their current registration.

    This is ANTs SyN (symmetric normalization; Avants et al. 2008, Medical Image
    Analysis), the most widely used diffeomorphic registration in neuroimaging,
    run by FireANTs (Jena et al. 2026, Nature Communications), which keeps SyN's
    model and runs its optimisation on the GPU.

    SyN bends the atlas onto a section image of each section, starting
    from the section's current registration (its linear placement, and its
    deformation when it has one), so each fit builds on the one before;
    undo goes back. Fit the whole section first, then refine regions with
    restrict_to.[[?landmark_triangulation  A piece displaced more than about 400 um from its
    place is moved with landmarks first.]] The section's marked damage regions are left out of
    the atlas automatically. Needs a position and a transform. Applied as the
    section's deformation, one undoable write; a later change to the
    section's position, orientation, cutting angles or transform clears it.

    Use restrict_to when a few clearly recognizable, surviving regions
    provide better registration evidence than the whole section, to refine
    the existing registration locally. Review the wider overlay afterward
    as well.

    Both images are scaled from their 1st to 99th percentile inside the
    atlas outline to 0-1. The step size and the number of iterations at each
    scale are fixed. The reply names the section image, atlas image and mask each
    fit used.

    Args:
        sections: One to twenty filenames (the extension may be left off); the
            fits run one after another.
        restrict_to: Regions to fit by (acronyms, names or ids, descendants
            included; "CTX:left" or "CTX:right" for one side): the atlas and
            the section are both cut to the regions' neighbourhood and the
            displacement fades to nothing beyond it. Only the ANTs and
            Elastix fits take it. Empty fits by every region.
        section_image: The section image to match, by name: "preprocessed", a raw channel~
            [[?make_image , or an image made with make_image]].
        atlas_image: One of the atlas's images, by name (the job statement lists
            them and says what each is), or "nissl", the atlas's Nissl image
            where it has one, else its template.
        mask: "none" compares every pixel of the section[[?make_mask , or the
            name of a mask of that section made with make_mask: pixels with
            a weight under 0.5 are left out]].
        smoothness_um: The warp's smoothness scale in micrometres, as for an
            adult mouse brain (a larger or smaller brain scales it): the
            width over which each update of the displacement field is
            smoothed, the whole field a quarter of that. Larger bends less.
        metric: "mutual_information" or "local_correlation": how the two
            images are compared.
        edges: True also matches the edges (the gradient magnitude) of both
            images, as a second term of equal weight.
        detail: "coarse", "standard" or "fine": the resolution the fit works
            at (fine: 10 um for an adult mouse brain) and the iterations it
            takes; finer is slower.
        view: True returns the picture described below; false returns none.

    Returns:
        Per section the displacement (median and max, mm, over the placed atlas),
        the fold fraction and plausibility flags (regions compressed,
        expanded, vanished or folded beyond limits; displacement outsized
        for the section[[?landmark_triangulation , measured beyond any landmark warp]]); the
        `inputs` it used (section image, atlas image as resolved, mask, damage
        regions left out, intensity scaling) and `estimated` (the fraction of
        the section the fit left out on its own); and a
        picture of each section under its new registration with the atlas borders on it. With
        restrict_to, those regions' borders are drawn thick and the others faint, and the
        picture is zoomed to them when they cover a small part of the
        section.
    """
    ...


def trace_borders(section: str, prompt: str = "", restrict_to: list[str] = []) -> dict[str, Any]:
    """Trace the atlas borders onto a section with the image model, then fit it with FireANTs.

    LangSlice's own nonlinear method, packaged: the image model is shown the
    section's "preprocessed" image with its placed atlas borders and draws
    them on the section's anatomy; when it answers, FireANTs fits the traced borders (80 um
    smoothness) on top of the section's current registration and the
    deformation is applied as its own undoable step. The section needs a
    position and a transform; its marked damage regions are left out of
    what the model is shown. With restrict_to, the model is shown only those
    regions' borders and only they are fitted. The first answer at a
    placement and region choice is saved and reused, whatever the prompt;
    when its fit is already applied, nothing is fitted again. Every prompt
    sent is saved in the job folder.

    A focused trace can address one difficult region without tracing the
    whole section. Use restrict_to to select it and, when helpful, tailor
    the full image prompt to that section while preserving the stated
    border-drawing rules. The tool handles both tracing and fitting the result.

    Starts the work in the background and returns at once with its id, so
    you can carry on with other sections. When the work finishes, the next
    tool reply starts with its notice: the fit's numbers, then its
    pictures (the fitted borders, and the trace on the section), each with
    its number and caption. status
    lists the work still running; submit waits for it. A write to this
    section is refused until the work lands.

    Args:
        section: Filename (the extension may be left off).
        prompt: The full image prompt for this section: the base prompt
            below, edited for this section; empty sends the base prompt.
            Rules for any edit: the atlas borders shown to the model are the
            only source of which lines exist, so its answer holds each of
            them and no other line, except where tissue is physically missing
            from the section; a faint or indistinct boundary is still drawn,
            where the atlas places it. Never add a sentence that makes a
            line depend on whether its edge is visible.
        restrict_to: Regions to trace and fit alone (acronyms, names or ids,
            descendants included; "CTX:left" or "CTX:right" for one side).
            Empty: every region.

    The base prompt (its image numbers are the images the image model
    receives):

    BASE_PROMPT
    """
    ...


# --- bookkeeping --------------------------------------------------------------------


def note(text: str) -> dict[str, Any]:
    """Append one line to the run notes, which are saved with the results.

    Args:
        text: The note.
    """
    ...


def undo(steps: int = 1) -> dict[str, Any]:
    """Undo the last write. A tool call, or a finished piece of background work, is one step.

    Args:
        steps: How many steps to undo at once, newest first (all the history
            holds when it holds fewer).

    Returns:
        The sections undone and the fields that changed, and the rows of
        those sections.
    """
    ...


def redo(steps: int = 1) -> dict[str, Any]:
    """Redo the write that ``undo`` reversed.

    Args:
        steps: How many undone steps to redo at once.

    Returns:
        The sections redone and the fields that changed, and the rows of
        those sections.
    """
    ...


def restore_registration(section: str, registration: int = 0,
                         view: bool = True) -> dict[str, Any]:
    """Put back a registration a section held before: its orientation, transform and deformation.

    Each registration a section holds gets a number, kept whatever is
    written after it: the `registration` of its status row and of the reply
    to the write that left it. Undo takes back the job's last write; this
    returns one section to any earlier registration, such as a fit made
    before another method's, and leaves the other sections as they are. One
    undoable write. A deformation fitted with the section at another position
    or other cutting angles is refused until the section is back there.

    Args:
        section: Filename (the extension may be left off).
        registration: The number to put back; 0 lists the section's
            registrations (number, the tool that made it, when) and writes
            nothing.
        view: Return a picture of the section under the registration put
            back.

    Returns:
        The registration the section holds now and the one it replaced, its
        row, and the picture; or the list.
    """
    ...


def submit(
    summary: str,
    notes: list[str],
    interval_breaks: list[str],
) -> dict[str, Any]:
    """End the run. Call this exactly once, last.

    Waits for the background work still running. Refused, with the reason
    and nothing changed, while something the run needs is missing.

    Args:
        summary: One or two sentences on what you did.
        notes: Short observations worth carrying forward.
        interval_breaks: Filenames (the extension may be left off) of the
            sections AFTER a gap you conclude is real. Empty if there are
            none.
    """
    ...


# --- for scripts only -----------------------------------------------------------------


def export_maps(slices: list[str] = [], full_resolution: bool = False) -> dict[str, Any]:
    """Write each placed section's maps and the stack's exports, from the job as it stands.

    Per section (sections/<name>/): coords.tif (atlas micrometres per
    pixel, float32, NaN outside the tissue or the atlas), labels.tif (atlas
    ids, uint32), labels_fiji.tif + labels.csv (a uint16 index Fiji opens
    without loss, and its table), residual.tif (the deformation, when one
    is applied) and maps.json. Then exports/quicknii.json and
    exports/visualign.json, and registration.json again. Only this writes
    the per-section maps (submit writes the exports alone); nothing in the
    state changes.

    Args:
        slices: Filenames (the extension may be left off); empty is every
            section.
        full_resolution: Write the maps on the image file's own pixels
            instead of its working copy (what every picture[[?zoom  but a zoom]] is
            drawn from).

    Returns:
        written (the sections), skipped (each with its reason), files
        (path and kind), seconds.
    """
    ...



def set_landmarks(section: str, pairs: list[LandmarkPair] = [],
                  delete: list[str] = [], view: bool = True) -> dict[str, Any]:
    """Add, update or delete persistent corresponding landmarks. Does not apply a warp.

    When a piece of tissue lies away from its place in the atlas outline (a
    cerebellum or olfactory bulb pulled off across a gap, a shifted
    hemisphere, a folded-back flap), move it with landmarks before a dense
    fit: pair points on the piece with their places in the atlas, apply them
    with landmark_triangulation[[?fireants_syn , then refine with fireants_syn, which
    keeps the landmark move and finishes the fit. Rough pairs are enough.
    fireants_syn alone moves such a piece only part of the way and stretches
    borders across the gap. A piece within about 400 um of its place needs no
    pairs: fireants_syn fits it alone]].

    Build corresponding pairs across the full section and atlas before calling
    landmark_triangulation to apply the set. Pair IDs are stable. Every batch
    is one undo step. Atlas and section endpoints are explicitly named.
    Coordinates are x,y PIXELS from the top-left of the whole returned picture,
    including panel offsets; no fractions or atlas coordinates are accepted.
    [[?zoom On a zoom, x,y are the numbers its grid prints (pixels of the whole
    picture it was cut from), so a number read off any grid works whether you
    name the zoom's picture or the whole picture. ]]Saved geometry resolves the
    exact referenced view, including old zooms, orientation and an applied warp.
    Caption/gutter/out-of-image points are refused, and so is an atlas endpoint
    outside the atlas (the refusal gives the nearest point inside it).
    [[?look:atlas A picture of the
    atlas alone supplies atlas points only when drawn with
    look(mode="atlas", sections=[section]); generic atlas browsing has no
    section context. ]]Plane changes retain pairs but flag their atlas
    correspondences stale; update those atlas endpoints.
    [[?look:overlay Prefer both endpoints on the same overlay[[?zoom , or a zoom of it]]: the
    atlas point where an atlas border is drawn, the section point where that
    border belongs in the tissue. You point at what you see; a point on a warped
    overlay is mapped back to the section itself. ]][[?zoom Read endpoints off a close
    zoom, its grid lines 10 to 20 pixels apart: read off a coarse grid, a point can
    miss the border you meant by 20 pixels or more. ]]Spread pairs over the outline and
    the internal anatomy: a pair on the outline moves the outline, and a pair
    on an internal landmark (the midline, a ventricle, a fibre tract) moves the
    anatomy inside. A pair moves the tissue around it, out to the neighbouring
    pairs and the image corners, so tissue that is already right takes pairs
    too: a pair with both endpoints on the same point[[?look:overlay  (one pixel
    of an overlay)]] keeps that spot where it is. [[?mark_damage  An atlas region
    with no tissue on the section (torn or missing) takes no pair: mark it with
    mark_damage; a region
    only partly missing keeps its pairs on the part that is present. ]]Tissue with no
    atlas counterpart (debris, or tissue from another section) takes no pair either.
    With no pairs and no delete, nothing is written and the reply lists the pairs.

    Args:
        section: Section filename (the extension may be left off).
        pairs: New pairs: {atlas: {picture, x, y}, section: {picture, x, y}}.
            Updates add id and either or both endpoints; omitted endpoints stay.
            At most four added/edited pairs with view true.
        delete: Stable pair IDs to delete from this section, or ["all"] to
            delete every pair of this section.
        view: Show one contextual side-by-side crop for each added/edited pair:
            atlas TEMPLATE and section PREPROCESSED. False shows no pictures.

    Returns:
        The changed and deleted IDs; for each pair given an atlas endpoint, the
        region drawn there (`atlas_endpoint_regions`); and every pair of the section, one record
        each: id, atlas and section as {picture, x, y}, and state: applied (in
        the landmark warp applied now, as they are), moved (in it, but at other
        endpoints now), pending (saved, not in it) or stale (placed against an
        earlier atlas plane); with the count of each; and `warp`: the pairs the
        current warp was built from, those not in it, those deleted but still
        shaping it, and that landmark_triangulation must run again to apply the
        set as it stands.
    """
    ...


def landmark_triangulation(
    section: str, view: bool = True, resolution: int = 0
) -> dict[str, Any]:
    """Apply this section's saved landmark set as a piecewise affine warp.

    Requires at least three corresponding pairs, with coverage across the full
    section; the tool cannot assess anatomical coverage. Each pair moves the
    tissue around it, out to the neighbouring pairs and the image corners, so
    place pairs on tissue that is already right as well, not only on tissue
    that must move: a pair with both endpoints on the same point of an overlay
    keeps that spot where it is now (its current placement: linear, or the
    earlier deformation this composes onto). Uses VisuAlign's section-domain
    Delaunay mesh in pixels over the pairs, and four fixed corners 10% outside
    the section. Repeated section points, zero-area triangles and stale atlas
    planes are refused. Folded triangles are applied, as VisuAlign does: the
    reply gives the share of the section they cover and the pairs at their
    corners, and the picture outlines them in magenta. The inverse uses the
    same triangles (unfolded ones first where folded ones overlap). Composes a
    correction onto the current deformation; undo removes the application.
    Editing pairs alone never warps: pairs added, moved or deleted after an
    application change nothing until this runs again.

    Args:
        section: Section filename (the extension may be left off).
        view: Show the resulting registration under atlas borders.
        resolution: The long edge of the picture in pixels, as for look; 0 is
            the default size.

    Returns:
        Landmark count, `built_from` (the pair IDs the warp was made of),
        whether applied/composed,
        displacement_mm (the whole deformation now applied) and
        step_displacement_mm (only the change this application made over the
        warp it replaced), diagnostics and any folds. The picture's
        number is under `pictures`.
    """
    ...

#: Every declared verb, in :data:`langslice.ops.registry.VERBS` order.
STUBS: dict[str, Callable[..., Any]] = {
    stub.__name__: stub for stub in (
        look, zoom, make_image,
        grep_atlas, grep_atlas_view, status, list_files, search_files, read_file,
        position_sections, interactive_transform, mark_damage, make_mask,
        elastix_affine, fireants_syn, set_landmarks, landmark_triangulation,
        trace_borders,
        note, undo, redo, restore_registration, submit, export_maps,
    )
}

#: The change tools: each returns its picture unless ``view`` is false.
VIEW_TOOLS: tuple[str, ...] = (
    "make_image", "make_mask", "position_sections", "interactive_transform", "elastix_affine",
    "fireants_syn", "set_landmarks", "landmark_triangulation", "restore_registration",
)

# --- the run's variant ------------------------------------------------------------


def base_prompt(plane: str = "coronal", provider: str | None = None) -> str:
    """The base image prompt ``trace_borders`` describes for *plane* and the
    image *provider* (``core.nonlinear.prompts``)."""
    from typing import cast

    from langslice.core.nonlinear.prompts import border_correction_tool_prompt
    from langslice.core.space import Plane

    return border_correction_tool_prompt(cast(Plane, plane), provider=provider)


@dataclass(frozen=True)
class Variant:
    """What of a run changes its declarations (see the module text)."""

    #: The caller chooses each picture's size (``look``'s ``resolution``).
    auto: bool = False
    #: Who reads the declaration (:data:`DOORS`): how a background verb answers.
    door: str = "agent"
    #: The host forces the change tools' pictures on: no ``view`` argument.
    forced_view: bool = False
    #: ``position_sections`` takes ``sections`` (the Positioning task is on).
    positions: bool = True
    #: The door puts no per-call cap on pictures or sections (the agent CLI and
    #: the library; ``Job.picture_cap`` None): :data:`_UNCAPPED_DOCS`.
    uncapped: bool = False
    #: The base image prompt ``trace_borders`` carries.
    prompt: str = ""
    #: What the job disables (``JobSpec.disabled``).
    off: frozenset[str] = frozenset()
    #: The tools the run lacks because their task is off. A tool plugin's text
    #: renders by these as well as by ``off`` (a plugin sentence that needs
    #: ``elastix_affine`` stands only where the Linear task is on); the native
    #: declarations keep their pinned wording and render by ``off`` alone.
    absent: frozenset[str] = frozenset()
    #: The tool plugins the job loads (:class:`langslice.core.plugins.LoadedPlugin`,
    #: compared by identity): their tools are declared from their own stubs.
    #: Empty: no plugin text is in any declaration.
    plugins: tuple[Any, ...] = ()
    #: The job's atlas brain's size next to the adult mouse's
    #: (``core.atlas.size.brain_scale``): the distances a description states
    #: (:data:`_SCALED_DOCS`) are the adult mouse's times it.
    brain_scale: float = 1.0

    @classmethod
    def of(cls, spec: Any, *, auto: bool, image_model: bool = True, door: str = "agent",
           prompt: str | None = None, uncapped: bool | None = None,
           brain_scale: float = 1.0) -> Variant:
        """The variant of a :class:`~langslice.core.spec.JobSpec`'s run;
        *image_model* False: the door cannot reach its image model, so no
        prompt is described. *prompt* is the image model's own base prompt
        (None: the job's own, ``nonlinear.image_prompt``, else LangSlice's
        for the run's plane and provider). *uncapped* (None: the CLI door
        is) as :attr:`uncapped`; *brain_scale* (the job's workspace's) as
        :attr:`brain_scale`."""
        traced = bool(spec.has("nonlinear") and spec.nonlinear.uses_image_model and image_model)
        text = ""
        if traced:
            own = spec.nonlinear.image_prompt
            if prompt is not None:
                text = prompt
            elif own is not None:
                from langslice.core.nonlinear.registration_tool import PLANE_PLACEHOLDER

                text = own.replace(PLANE_PLACEHOLDER, str(spec.plane))
            else:
                text = base_prompt(str(spec.plane), spec.nonlinear.provider)
        from langslice.core.plugins import of_spec
        from langslice.core.toolset import tools_for

        held = tuple(of_spec(spec))
        absent = frozenset(name for name, tool in tools_for(spec).items()
                           if not tool.when(spec)) if held else frozenset()
        return cls(auto=bool(auto), door=door, forced_view=bool(spec.force_view), absent=absent,
                   uncapped=door == "cli" if uncapped is None else uncapped,
                   positions=bool(spec.has("position")),
                   prompt=text, off=frozenset(spec.switched_off), plugins=held,
                   brain_scale=float(brain_scale))


#: The doors a verb is declared for: ``agent`` (LangSlice's own agent, and
#: the library), ``mcp`` (Claude Desktop) and ``cli`` (the agent CLI).
DOORS = ("agent", "mcp", "cli")

#: Passages each door words its own way, as ``(agent text, {door: text})``
#: per verb: how a background verb answers (the CLI waits for its work to
#: land unless run with ``--background``).
_DOOR_DOCS: dict[str, tuple[tuple[str, dict[str, str]], ...]] = {
    "trace_borders": ((
        "Starts the work in the background and returns at once with its id, so\n"
        "        you can carry on with other sections. When the work finishes, the next\n"
        "        tool reply starts with its notice:",
        {"cli": "The command answers once the work has finished (with --background\n"
                "        at once, and `wait` collects the answer), with its notice:"},
    ),),
}


#: Passages worded for a door with no picture or section cap
#: (:attr:`Variant.uncapped`), as ``(capped text, uncapped text)`` per verb.
_UNCAPPED_DOCS: dict[str, tuple[tuple[str, str], ...]] = {
    "look": ((
        "Up to 4 pictures, attached in the order `pictures` lists them, each\n"
        "with its number and caption. Pictures past 4 are listed under\n"
        "`not_shown` with the call that draws them.",
        "Every picture drawn, in the order `pictures` lists them, each with its\n"
        "number and caption."),),
    "make_image": ((
        "up to 4 of the sections (spread over the stack when none are named)",
        "each of the sections (the whole stack when none are named)"),),
    "make_mask": ((
        "up to 4 of the\nsections (spread over the stack when none are named)",
        "each of the sections\n(the whole stack when none are named)"),),
    "grep_atlas_view": ((
        "Up to 4 pictures with their numbers and captions",
        "One picture per position with its number and caption"),),
    "interactive_transform": (("sections: One to four objects:",
                               "sections: One or more objects:"),),
    "elastix_affine": (("sections: One to four filenames", "sections: One or more filenames"),),
}


#: How far a displaced piece lies from its place before a dense fit alone no
#: longer reaches it, so landmarks move it first (adult mouse; sagittal and
#: horizontal sections, 2026-10-17).
LANDMARK_MOVE_UM = 400.0

#: Distances a description states, as ``(text, adult mouse micrometres)`` per
#: verb: worded for a job whose atlas brain is another size
#: (:attr:`Variant.brain_scale`), the number times the scale.
_SCALED_DOCS: dict[str, tuple[tuple[str, float], ...]] = {
    "elastix_affine": (("atlas within 300 um of them", DEFAULT_NEIGHBOURHOOD_UM),),
    "fireants_syn": (("more than about 400 um from its", LANDMARK_MOVE_UM),),
    "set_landmarks": (("within about 400 um of its", LANDMARK_MOVE_UM),),
}


def _scaled_doc(name: str, doc: str, scale: float) -> str:
    """*doc* with the distances :data:`_SCALED_DOCS` names at brain *scale*."""
    for text, um in _SCALED_DOCS.get(name, ()):
        doc = _reworded(name, doc, text, text.replace(f"{um:g} um", f"{um * scale:.0f} um"))
    return doc


def _reworded(name: str, doc: str, capped: str, uncapped: str) -> str:
    """*doc* with the passage *capped* (its line breaks and indentation as
    they fall) replaced by *uncapped*, indented as the passage's line is."""
    pattern = r"\s+".join(re.escape(word) for word in capped.split())
    found = re.search(pattern, doc)
    if found is None:
        raise RuntimeError(f"{name}: the passage a variant rewords is gone")
    line = doc.rfind("\n", 0, found.start()) + 1
    indent = doc[line:len(doc) - len(doc[line:].lstrip(" "))]
    return doc[:found.start()] + uncapped.replace("\n", "\n" + indent) + doc[found.end():]


def _door_doc(name: str, doc: str, door: str) -> str:
    """*doc* worded for *door* (:data:`_DOOR_DOCS`)."""
    for agent_text, worded in _DOOR_DOCS.get(name, ()):
        if door in worded:
            if agent_text not in doc:
                raise RuntimeError(f"{name}: the passage {door!r} rewords is gone")
            doc = doc.replace(agent_text, worded[door])
    return doc


#: The variant a caller without a job sees: every argument, LangSlice's base
#: prompt for coronal sections and the OpenAI image models. The CLI's
#: ``langslice-job schema`` uses it with ``auto`` (the CLI's pictures are
#: sized by the caller).
FULL = Variant(prompt=base_prompt("coronal", "openai-oauth"))

#: Where the base prompt goes in ``trace_borders``' description.
_PROMPT_MARK = "BASE_PROMPT"


def _without_argument(doc: str, name: str) -> str:
    """*doc* without the ``Args`` entry of argument *name* (its first line
    and every more deeply indented line after it)."""
    lines = doc.split("\n")
    head = " " * 12 + f"{name}:"
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith(head):
            skipping = True
            continue
        if skipping and line.startswith(" " * 16):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


def argument_texts(doc: str) -> dict[str, str]:
    """Each argument's ``Args`` entry of a declared description *doc* (as
    :func:`model_doc` indents it), on one line: ``{name: text}``."""
    texts: dict[str, str] = {}
    name = None
    for line in doc.split("\n"):
        if line.startswith(" " * 16) and name is not None:
            texts[name] += " " + line.strip()
            continue
        name = None
        head = line[12:] if line.startswith(" " * 12) and not line.startswith(" " * 13) else ""
        key, colon, text = head.partition(":")
        if colon and key.isidentifier():
            name = key
            texts[name] = text.strip()
    return texts


def _with_prompt(doc: str, prompt: str) -> str:
    """``trace_borders``' description with the base prompt in place of its
    mark, each line indented as the description is (none without a prompt)."""
    indent = " " * 8
    if not prompt:
        cut = doc.index("\n\n" + indent + "The base prompt")
        end = doc.index(_PROMPT_MARK) + len(_PROMPT_MARK)
        return doc[:cut] + doc[end:]
    body = "\n".join((indent + line) if line else "" for line in prompt.split("\n"))
    return doc.replace(indent + _PROMPT_MARK, body)


# --- declarations -------------------------------------------------------------------


def model_doc(stub: Callable[..., Any]) -> str:
    """*stub*'s docstring as a model reads it: continuation lines indented by
    eight spaces (see the module text)."""
    lines = (stub.__doc__ or "").split("\n")
    return "\n".join([lines[0], *[("    " + line) if line else line for line in lines[1:]]])


@dataclass(frozen=True)
class Declaration:
    """One verb as a door offers it: name, description, arguments."""

    name: str
    doc: str
    signature: inspect.Signature
    annotations: dict[str, Any]

    @property
    def summary(self) -> str:
        """The description's first line."""
        return self.doc.split("\n", 1)[0].strip()


def dropped_arguments(name: str, variant: Variant) -> tuple[str, ...]:
    """The arguments of verb *name* a run of *variant* does not offer."""
    dropped: list[str] = []
    if name in ("look", "landmark_triangulation") and not variant.auto:
        dropped.append("resolution")
    if name in VIEW_TOOLS and variant.forced_view:
        dropped.append("view")
    if name == "position_sections" and not variant.positions:
        dropped.append("sections")
    if name in ("look", "elastix_affine", "fireants_syn") and is_off(variant.off, "make_mask"):
        dropped.append("masks" if name == "look" else "mask")  # nothing to name
    if name == "fireants_syn":
        dropped += [option for option in TOOLS[name].options
                    if is_off(variant.off, f"{name}{OPTION_SEPARATOR}{option}")]
    if name == "look":
        modes = set(options_on(variant.off, "look"))
        if "overlay" not in modes:  # what only an overlay uses
            dropped += ["atlas_opacity", "warp"]
        if not modes & {"atlas", "positioning"}:  # the atlas planes drawn
            dropped.append("positions_mm")
        if not modes & {"overlay", "atlas", "positioning"}:  # an atlas to layer
            dropped.append("atlas_layers")
    if name == "look" and not set(options_on(variant.off, "look")) & {"section", "overlay"}:
        dropped.append("layers")  # they are drawn on those two modes only
    if name == "grep_atlas" and is_off(variant.off, "grep_atlas:points"):
        dropped.append("points")
    if name == "make_image" and is_off(variant.off, "look:section"):
        dropped.append("view")  # its pictures are look's section pictures
    if name == "zoom" and is_off(variant.off, "look:overlay"):
        dropped.append("region")  # a region is zoomed on an overlay only
    return tuple(dropped)


def narrowed(name: str, argument: str, annotation: Any, off: Collection[str]) -> Any:
    """*annotation* of verb *name*'s *argument* without the options the job
    disables: the values of an enumeration (``look``'s ``mode``), the
    ``flip`` and ``rotate_quarter`` keys of ``interactive_transform``'s sections."""
    options = TOOLS[name].options if name in TOOLS else {}
    if argument in options.values():
        values = typing.get_args(annotation)
        return Literal[tuple(value for value in values
                             if not is_off(off, f"{name}{OPTION_SEPARATOR}{value}"))]
    if (name, argument) == ("interactive_transform", "sections"):
        no_flip = is_off(off, "interactive_transform:flip")
        no_rotate = is_off(off, "interactive_transform:rotate")
        if no_flip and no_rotate:
            return list[SectionTransformNoOrientation]
        if no_flip:
            return list[SectionTransformNoFlip]
        if no_rotate:
            return list[SectionTransformNoRotate]
    return annotation


def plugin_stub(name: str, variant: Variant) -> Callable[..., Any]:
    """The stub of tool plugin tool *name*, from the plugins *variant* holds."""
    for plugin in variant.plugins:
        for tool in plugin.tools:
            if tool.name == name:
                return tool.stub  # type: ignore[no-any-return]
    raise KeyError(name)


@functools.cache
def declaration(name: str, variant: Variant = FULL) -> Declaration:
    """The declaration of verb *name* in a run of *variant*."""
    stub = STUBS[name] if name in STUBS else plugin_stub(name, variant)
    signature = inspect.signature(stub, eval_str=True)
    annotations = dict(inspect.get_annotations(stub, eval_str=True))
    doc = _door_doc(name, model_doc(stub), variant.door)
    if variant.uncapped:
        for capped, uncapped in _UNCAPPED_DOCS.get(name, ()):
            doc = _reworded(name, doc, capped, uncapped)
    if variant.brain_scale != 1.0:
        doc = _scaled_doc(name, doc, variant.brain_scale)
    doc = render(doc, variant.off if name in STUBS else variant.off | variant.absent)
    if name == "trace_borders":
        doc = _with_prompt(doc, variant.prompt)
    dropped = dropped_arguments(name, variant)
    for argument in dropped:
        doc = _without_argument(doc, argument)
        annotations.pop(argument, None)
    if variant.off:
        annotations = {argument: narrowed(name, argument, annotation, variant.off)
                       for argument, annotation in annotations.items()}
    parameters = [p.replace(annotation=annotations.get(p.name, p.annotation))
                  for p in signature.parameters.values() if p.name not in dropped]
    return Declaration(name, doc, signature.replace(parameters=parameters), annotations)


def summary(name: str, variant: Variant = FULL) -> str:
    """Verb *name*'s one-line description (a plugin tool's: in *variant*, which
    holds its plugin)."""
    return declaration(name, variant).summary


def declare(name: str, body: Callable[..., Any], variant: Variant = FULL) -> Callable[..., Any]:
    """Tool body *body* as verb *name* is declared in *variant*'s run.

    The returned function carries the declaration's name, description and
    signature (what ADK and FastMCP read); a call binds its arguments to that
    signature, fills the declared defaults and hands every argument to
    *body* by name. *body* takes every declared argument (a missing one is a
    ``TypeError`` here, when the tools are built); a parameter of *body* that
    is not declared (a door-only one, or one this run drops) keeps *body*'s
    default, except ADK's ``tool_context``, which is added to the signature
    so ADK passes it.
    """
    declared = declaration(name, variant)
    taken = inspect.signature(body).parameters
    missing = [arg for arg in declared.signature.parameters if arg not in taken]
    if missing:
        raise TypeError(f"{name}: the tool body takes no {', '.join(missing)}")
    signature = declared.signature
    annotations = dict(declared.annotations)
    if "tool_context" in taken:
        signature = signature.replace(parameters=[
            *signature.parameters.values(),
            inspect.Parameter("tool_context", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None, annotation=Any),
        ])
        annotations = {**{k: v for k, v in annotations.items() if k != "return"},
                       "tool_context": Any, "return": annotations.get("return")}

    def tool(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return body(**bound.arguments)

    tool.__name__ = tool.__qualname__ = name
    tool.__doc__ = declared.doc
    tool.__signature__ = signature  # type: ignore[attr-defined]
    tool.__annotations__ = annotations
    return tool


def arguments_schema(name: str, variant: Variant = FULL) -> dict[str, Any]:
    """The JSON schema of verb *name*'s arguments (an object; unknown keys
    refused, as every door refuses them), built by pydantic from the
    declaration, the way FastMCP builds its input schemas."""
    from pydantic import ConfigDict, create_model

    declared = declaration(name, variant)
    fields: dict[str, Any] = {}
    for parameter in declared.signature.parameters.values():
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (parameter.annotation, default)
    model = create_model(f"{name}_arguments", __config__=ConfigDict(extra="forbid"),
                         **fields)
    schema = model.model_json_schema()
    schema["title"] = name
    schema["description"] = declared.summary
    return schema
