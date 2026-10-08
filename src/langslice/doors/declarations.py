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
:func:`model_doc` adds the four this module level lacks (the goldens pin the
bytes).

A run varies a declaration in the ways a :class:`Variant` names: ``look``'s
``resolution`` exists only where the agent chooses the picture size (image
resolution "auto"; the CLI always); the change tools' ``view`` is not
offered where the host forces their pictures on (``JobSpec.force_view``);
``position_sections`` takes ``sections`` only with the Positioning task;
``submit`` takes ``left_linear`` only with Nonlinear on and no deformation
required on every section; ``trace_borders`` carries the run's base image
prompt; and each door words where its answers come from (``_DOOR_DOCS``).
An argument a run drops leaves the description with its entry.

Door-only parameters (ADK's ``tool_context``) are the tool body's, never
declared. Nothing here imports ``google.*``, ``litellm`` or ``openai``.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langslice.core import appearance as looks
from langslice.doors.tools.arguments import (
    CuttingAngles,
    LeftLinear,
    SectionPosition,
    SectionTransform,
)

# Defaults of list and object arguments are read, never mutated (ADK wants a value).
# ruff: noqa: B006

#: The choices a string argument takes, offered as an enum in every door's schema.
LookMode = Literal["section", "atlas", "overlay", "positioning"]
AtlasLayer = Literal["template", "nissl", "borders"]
Warp = Literal["applied", "none"]
FitAtlas = Literal["template", "nissl"]
Stiffness = Literal["soft", "medium", "firm"]

# --- looking ---------------------------------------------------------------------


def look(
    mode: LookMode,
    sections: list[str] = [],
    positions_mm: list[float] = [],
    channels: list[str] = [],
    atlas_layers: list[AtlasLayer] = [],
    atlas_opacity: float = 0.5,
    warp: Warp = "applied",
    resolution: int = 0,
) -> dict[str, Any]:
    """Draw pictures of sections and the atlas. Writes nothing.

    The same request draws the same kind of picture at every stage of the
    run: no default depends on the stack's state. Each picture gets a number
    and a caption stating the positions, cutting angles, scale and channel
    settings it was drawn with; zoom takes the number.

    Args:
        mode: "section": each section alone, as oriented, framed to its
            tissue. "atlas": the atlas plane alone at each of positions_mm,
            at the stack's cutting angles. "overlay": each section under its
            current registration, with the atlas borders and any other
            atlas_layers on it. "positioning": the sections and the atlas
            along the slicing axis as ABBA lays them out: the atlas at
            positions_mm above a millimetre ruler, the sections below it in
            order of position, each joined by a line to its position; a long
            stack is split into several pictures, none shrunk.
        sections: Filenames (the extension may be left off); empty is every
            section.
        positions_mm: Atlas positions in millimetres along the slicing axis:
            the planes "atlas" draws (at least one), and the atlas pictures
            above the ruler in "positioning" (none when empty).
        channels: The raw channels of each section to draw, by name (several
            are blended in distinct colours, at most 6), or ["preprocessed"]
            for the channel the fits and the image model read. Empty is every
            raw channel, each with its display settings.
        atlas_layers: Any of "template", "nissl" (Allen mouse atlases only)
            and "borders". Empty is the mode's default: borders in "overlay",
            template in "atlas" and "positioning". "section" draws no atlas.
        atlas_opacity: 0 to 1: how strongly an atlas image (template or
            nissl) shows under the section in "overlay".
        warp: "applied" draws each section with its deformation, "none" with
            its linear placement only.
        resolution: The long edge of each picture in pixels; 0 is the
            default size. A picture is never drawn larger than its source.

    Returns:
        Up to 4 pictures, attached in the order `pictures` lists them, each
        with its number and caption. Pictures past 4 are listed under
        `not_shown` with the call that draws them.
    """
    ...


def zoom(box: list[float], picture: int = 0) -> dict[str, Any]:
    """Draw a box of an earlier picture again, in more detail. Writes nothing.

    The new picture is as large as the earlier one. In a picture of a
    section or of a section with the atlas on it, the section is read again
    from its original image file at the file's own resolution, only inside
    the box, and drawn with the same channels, display settings and atlas
    borders; the caption says when the file holds fewer pixels than the
    picture. A positioning picture's sections are drawn again from their
    working copies (the file reduced to at most 3072 pixels), and the atlas
    from its voxels. The box is drawn as the picture was, even if the stack
    has changed since (the reply then says `stale`). The new picture has its
    own number, so it can be zoomed again.

    Args:
        box: [x0, y0, x1, y1] in pixels of that picture as it was shown,
            from its top-left corner.
        picture: The picture's number; 0 is the newest picture that is not
            itself a zoom.

    Returns:
        One picture with its number and caption, and `redrawn`: false when
        the picture could not be drawn again, so the box was cut from the
        saved picture and enlarged, with no more detail than it had.
    """
    ...


def set_channel_properties(
    channel: str,
    contrast_limits: list[float] = [],
    gamma: float = 0.0,
    colormap: str = "",
    reset: bool = False,
) -> dict[str, Any]:
    """Set how a raw channel is displayed, in every section that has it.

    Display only: it changes how look draws the channel and nothing a fit
    or the image model reads; the image files are never edited. The setting
    stays until it is changed, and every caption states it. Undoable.

    Args:
        channel: The raw channel's name.
        contrast_limits: [low, high] in the file's own intensities: low is
            drawn black, high at full brightness. Empty keeps the current
            limits. Without limits a channel is drawn with automatic ones,
            per section: black at the background, full brightness at the
            tissue's 95th percentile.
        gamma: 0.1 to 10; 1 is linear. 0 keeps the current gamma.
        colormap: "gray", "red", "green", "blue", "magenta", "cyan" or
            "yellow"; empty keeps the current one.
        reset: True returns the channel to its default display; the other
            arguments are then ignored.

    Returns:
        The settings now in force, the sections that have the channel, and
        its file intensities over them: the sample type and its range, and
        the 1st and 99.5th percentiles.
    """
    ...


def set_preprocessed_channel_properties(
    sections: list[str] = [],
    channel_weights: list[float] = [],
    clahe_clip: float = looks.DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = looks.DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    """Set the recipe of the preprocessed channel that the fits and the image model read.

    Every section has one preprocessed channel, made from its raw channels
    by a recipe, LangSlice's default until this is called: each raw channel
    with a weight above zero is optionally N4-corrected and denoised (ANTs)
    and contrast-enhanced by CLAHE, then the channels are blended by their
    weights. The fits match it against an atlas image. The raw channels are
    never changed; look draws this channel with channels ["preprocessed"].
    Undoable; another call replaces the recipe.

    Args:
        sections: Filenames (the extension may be left off) of the sections
            that get a recipe of their own; empty sets the whole stack's
            recipe.
        channel_weights: One weight per raw channel, in the order the reply's
            `channels` lists them; 0 leaves a channel out. Empty: automatic
            weights by tissue coverage.
        clahe_clip: CLAHE clip limit, 0 (no CLAHE) to 40.
        clahe_tiles: CLAHE tiles per side, 1 to 32.
        n4: ANTs N4 correction of uneven illumination.
        denoise: ANTs denoising.
        reset: True returns the stack to the default recipe, or the named
            sections to the stack's recipe; the other arguments are then
            ignored.

    Returns:
        The recipe in force, the raw channels, and for up to 4 of the
        sections (spread over the stack when none are named) two pictures
        each: the preprocessed channel before this call, then after it.
    """
    ...


def grep_atlas(query: str, section: str = "") -> dict[str, Any]:
    """Look regions up in the atlas hierarchy by acronym, name or id. Text only.

    To see a region's borders on the atlas, use grep_atlas_view.

    Args:
        query: Text matched case-insensitively against region acronyms and
            names (substring), or an exact acronym or numeric id.
        section: Optional filename (the extension may be left off) of a
            section with a position; each row then says whether the region
            (or any descendant) appears in the atlas plane at that section's
            placement.

    Returns:
        Rows of acronym, id, name, ancestry (root to parent, as acronyms)
        and descendant count, at most 40, with the number left over.
    """
    ...


def grep_atlas_view(regions: list[str], positions_mm: list[float]) -> dict[str, Any]:
    """Draw the atlas at given positions with chosen regions' borders highlighted. Writes nothing.

    The named regions' borders are drawn strong and the other borders
    faint, at the stack's cutting angles. Look highlights no regions; this
    tool does.

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
        One row per section in stack order: filename, position
        (position_source "default": a starting position the job gave the
        section, not yet placed), distance to the next placed section, flip
        and quarter turn, transform, damaged regions and damage note,
        deformation steps. Also the cutting angles, the interval breaks, the
        raw channels' display settings ("auto": automatic contrast limits),
        the preprocessed channel's recipe, the background work still
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
    number, tool, sections and caption), never its pixels: zoom shows it
    again by that number.

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
        call, the rows that changed, and a positioning picture of the
        written sections: the atlas at their new positions above a
        millimetre ruler, the sections below it, each joined by a line to
        its position.
    """
    ...


def interactive_transform(
    sections: list[SectionTransform],
    view: bool = True,
) -> dict[str, Any]:
    """Set sections' orientation and in-plane transform by hand, and show the result.

    For a few sections at a time, in a look-and-adjust loop. Values are
    absolute: a value given replaces the section's current one, and a value
    left out keeps it (a section without a transform starts from the
    identity). The quarter turn is applied first, then the flip. A changed
    orientation keeps the other values: the transform is rebuilt on the new
    orientation, and the reply says so. One undoable write; the image files
    are never modified.

    Whether a section is mirrored can only be decided from a visible notch,
    or from a cutting angle oblique enough that the two hemispheres'
    anatomy differs.

    Args:
        sections: One to four objects: "id" (filename; the extension may be
            left off) and any of "flip" (true mirrors the section
            left-right), "rotate_quarter" (0, 90, 180 or 270 degrees
            counter-clockwise), "rotation_deg" (degrees counter-clockwise
            about the pivot, the canvas centre),
            "scale_x" and "scale_y" (multipliers about the pivot; 1.0 keeps
            the size), "shear" (a slant applied before the rotation: each
            point moves sideways by shear times its distance below the pivot,
            in units of scale_x), "translate_x_mm" (right) and
            "translate_y_mm" (down). A section appears once per call.
        view: True returns the picture described below; false returns none.

    Returns:
        Per section its orientation, its transform as the values above,
        and whether anything was written; and a picture of each section
        under its new transform with the atlas borders on it.
    """
    ...


def mark_damage(section: str, regions: list[str], note: str = "") -> dict[str, Any]:
    """Mark the atlas regions a section has lost, so that every fit leaves them out.

    Damage is tissue that is missing from the section, or so badly
    displaced that it would wreck a fit, such as a whole hemisphere, the
    olfactory bulb or the cortex. Small tears, bubbles, stains, low contrast
    and folds within an intact outline are not damage and are not marked.
    A section is damaged exactly when it has marked regions. The marked
    regions are left out of elastix_affine, ants_syn and the image model's
    trace automatically, and they move with the registration because they
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


def elastix_affine(
    sections: list[str] = [],
    restrict_to: list[str] = [],
    atlas_image: FitAtlas = "template",
    view: bool = True,
) -> dict[str, Any]:
    """Fit sections' in-plane affine transforms automatically with elastix.

    Elastix refines each section's current placement (from the identity
    when it has no transform) by matching its preprocessed channel against
    an atlas image, inner anatomy included. It adjusts from where the
    section is and does not search from scratch: it moves a section by
    small amounts and does not turn it over. The section's marked damage
    regions are left out automatically. Each fit is written as the
    section's transform, one undoable write for the call.

    Args:
        sections: Filenames (the extension may be left off); empty is every
            placed section the user has not locked.
        restrict_to: Regions to fit by (acronyms, names or ids, descendants
            included; "CTX:left" or "CTX:right" for one side): only the
            atlas within 300 um of them, against the tissue the fit lays
            there. Empty fits by every region.
        atlas_image: "template" (the atlas's reference template) or "nissl"
            (a Nissl-stained reference, Allen mouse atlases only).
        view: True returns the picture described below; false returns none.

    Returns:
        Per section the overlap of the fitted shapes (iou; it compares
        outlines, not the anatomy inside, so a turned section can score as
        high as a correct one) and the transform (rotation_deg, scale_x,
        scale_y, shear, translate_x_mm, translate_y_mm about the canvas
        centre); and a picture of each fitted section
        under its new transform with the atlas borders on it. With
        restrict_to, those regions' borders are drawn thick and the others
        faint, and the picture is zoomed to them when they cover a small
        part of the section.
    """
    ...


def ants_syn(
    sections: list[str],
    restrict_to: list[str] = [],
    atlas_image: FitAtlas = "template",
    stiffness: Stiffness = "medium",
    view: bool = True,
) -> dict[str, Any]:
    """Fit a deformation of the atlas onto sections with ANTs SyN, on their current registration.

    SyN bends the atlas onto each section's preprocessed channel, starting
    from the section's current registration (its linear placement, and its
    deformation when it has one), so each fit builds on the one before;
    undo goes back. Fit the whole section first, then refine regions with
    restrict_to. The section's marked damage regions are left out
    automatically. Needs a position and a transform. Applied as the
    section's deformation, one undoable write; a later change to the
    section's position, orientation, cutting angles or transform clears it.

    Args:
        sections: One to four filenames (the extension may be left off).
        restrict_to: Regions to fit by (acronyms, names or ids, descendants
            included; "CTX:left" or "CTX:right" for one side). Empty fits by
            every region.
        atlas_image: "template" (the atlas's reference template) or "nissl"
            (a Nissl-stained reference, Allen mouse atlases only).
        stiffness: "soft", "medium" or "firm".
        view: True returns the picture described below; false returns none.

    Returns:
        Per section the displacement (median and max, mm, over the tissue),
        the fold fraction and plausibility flags (regions compressed,
        expanded, vanished or folded beyond limits; displacement outsized
        for the section); and a picture of each section under its new
        registration with the atlas borders on it. With restrict_to, those
        regions' borders are drawn thick and the others faint, and the
        picture is zoomed to them when they cover a small part of the
        section.
    """
    ...


def trace_borders(section: str, prompt: str = "", restrict_to: list[str] = []) -> dict[str, Any]:
    """Trace the atlas borders onto a section with the image model, then fit what it drew with ANTs.

    LangSlice's own nonlinear method, packaged: the image model is shown the
    section with its placed atlas borders and draws them on the section's
    anatomy; when it answers, ANTs fits the traced borders (medium
    stiffness) on top of the section's current registration and the
    deformation is applied as its own undoable step. The section needs a
    position and a transform; its marked damage regions are left out of
    what the model is shown. With restrict_to, the model is shown only those
    regions' borders and only they are fitted. The first answer at a
    placement and region choice is saved and reused, whatever the prompt;
    when its fit is already applied, nothing is fitted again. Every prompt
    sent is saved in the job folder.

    Starts the work in the background and returns at once with its id, so
    you can carry on with other sections. When the work finishes, the next
    tool reply starts with its notice: the fit's numbers, then its
    pictures (the fitted borders, and the trace on the section), each with
    its number and caption. status
    lists the work still running; submit waits for it.

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


def undo() -> dict[str, Any]:
    """Undo the last write. A tool call, or a finished piece of background work, is one step."""
    ...


def redo() -> dict[str, Any]:
    """Redo the write that ``undo`` reversed."""
    ...


def submit(
    summary: str,
    notes: list[str],
    interval_breaks: list[str],
    left_linear: list[LeftLinear] = [],
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
        left_linear: Sections left without a deformation, each with the
            reason its linear placement stands: [{"id": "<filename>",
            "reason": "..."}].
    """
    ...


# --- for scripts only -----------------------------------------------------------------


def trace_from_atlas(slices: list[str], passes: int = 1) -> dict[str, Any]:
    """Trace sections' atlas borders with the image model, shown no placement.

    The model sees each clean section and the outlined atlas plane at its
    position and cutting angles; passes=2 adds a corrective second call. The
    reply is recorded as trace_borders records its own; a script fits it
    with langslice.ops.deformable.fit_deformable (fit_section
    "traced_borders"), from the section's linear transform.

    Args:
        slices: Filenames (the extension may be left off), each with a
            position and linear transform.
        passes: 1, or 2 for the corrective second call.

    Starts the image calls in the background and returns at once; the
    results are saved, and submit waits for them. Does not fit a deformation.
    """
    ...


def export_maps(slices: list[str] = [], full_resolution: bool = False) -> dict[str, Any]:
    """Write each placed section's maps and the stack's exports, from the job as it stands.

    Per section (sections/<name>/): coords.tif (atlas micrometres per
    pixel, float32, NaN outside the tissue or the atlas), labels.tif (atlas
    ids, uint32), labels_fiji.tif + labels.csv (a uint16 index Fiji opens
    without loss, and its table), residual.tif (the deformation, when one
    is applied) and maps.json. Then exports/quicknii.json and
    exports/visualign.json, and registration.json again. submit writes the
    same files; nothing in the state changes.

    Args:
        slices: Filenames (the extension may be left off); empty is every
            section.
        full_resolution: Write the maps on the image file's own pixels
            instead of its working copy (what every picture but a zoom is
            drawn from).

    Returns:
        written (the sections), skipped (each with its reason), files
        (path and kind), seconds.
    """
    ...


#: Every declared verb, in :data:`langslice.ops.registry.VERBS` order.
STUBS: dict[str, Callable[..., Any]] = {
    stub.__name__: stub for stub in (
        look, zoom, set_channel_properties, set_preprocessed_channel_properties,
        grep_atlas, grep_atlas_view, status, list_files, search_files, read_file,
        position_sections, interactive_transform, mark_damage,
        elastix_affine, ants_syn, trace_borders, trace_from_atlas,
        note, undo, redo, submit, export_maps,
    )
}

#: The change tools: each returns its picture unless ``view`` is false.
VIEW_TOOLS: tuple[str, ...] = (
    "position_sections", "interactive_transform", "elastix_affine", "ants_syn",
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
    #: ``submit`` takes ``left_linear`` (Nonlinear on, no deformation required).
    left_linear: bool = True
    #: The base image prompt ``trace_borders`` carries.
    prompt: str = ""

    @classmethod
    def of(cls, spec: Any, *, auto: bool, image_model: bool = True, door: str = "agent",
           prompt: str | None = None) -> Variant:
        """The variant of a :class:`~langslice.core.spec.JobSpec`'s run;
        *image_model* False: the door cannot reach its image model, so no
        prompt is described. *prompt* is the image model's own base prompt
        (None: LangSlice's for the run's plane and provider)."""
        traced = bool(spec.has("nonlinear") and spec.nonlinear.uses_image_model and image_model)
        text = ""
        if traced:
            text = prompt if prompt is not None else base_prompt(
                str(spec.plane), spec.nonlinear.provider)
        return cls(auto=bool(auto), door=door, forced_view=bool(spec.force_view),
                   positions=bool(spec.has("position")),
                   left_linear=bool(spec.has("nonlinear")
                                    and not spec.nonlinear.require_deformation),
                   prompt=text)


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
    "trace_from_atlas": ((
        "Starts the image calls in the background and returns at once; the\n"
        "        results are saved, and submit waits for them.",
        {"cli": "Runs the image calls and answers once they have landed (with\n"
                "        --background at once; `wait` collects the answer); the results\n"
                "        are saved, and submit waits for them."},
    ),),
}


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
    if name == "look" and not variant.auto:
        dropped.append("resolution")
    if name in VIEW_TOOLS and variant.forced_view:
        dropped.append("view")
    if name == "position_sections" and not variant.positions:
        dropped.append("sections")
    if name == "submit" and not variant.left_linear:
        dropped.append("left_linear")
    return tuple(dropped)


@functools.cache
def declaration(name: str, variant: Variant = FULL) -> Declaration:
    """The declaration of verb *name* in a run of *variant*."""
    stub = STUBS[name]
    signature = inspect.signature(stub, eval_str=True)
    annotations = dict(inspect.get_annotations(stub, eval_str=True))
    doc = _door_doc(name, model_doc(stub), variant.door)
    if name == "trace_borders":
        doc = _with_prompt(doc, variant.prompt)
    dropped = dropped_arguments(name, variant)
    for argument in dropped:
        doc = _without_argument(doc, argument)
        annotations.pop(argument, None)
    parameters = [p for p in signature.parameters.values() if p.name not in dropped]
    return Declaration(name, doc, signature.replace(parameters=parameters), annotations)


def summary(name: str) -> str:
    """Verb *name*'s one-line description."""
    return declaration(name).summary


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
