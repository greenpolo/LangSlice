"""Every verb's declaration: the one home of what each door offers.

Each verb of
:data:`langslice.ops.registry.VERBS` is declared here ONCE, as a stub
function below: its arguments (names, types, defaults) and its description.
Every door is generated from these declarations:

- the agent tools and the MCP tools: ``langslice.doors.tools.toolbox.build_tools``
  puts each tool body behind :func:`declare`, so ADK's function declaration
  and FastMCP's input schema are read off these signatures and docstrings;
- the agent CLI: ``langslice ops`` (:func:`summary`), ``langslice schema
  VERB`` (:func:`arguments_schema`), ``langslice job FOLDER VERB``;
- the job folder's reference card and the library's job methods.

The description is the stub's docstring, which a model reads verbatim (ADK
and FastMCP send a function's ``__doc__`` as it is). Every continuation line
a model reads is indented eight spaces; :func:`model_doc` adds the four this
module level lacks (the goldens pin the bytes).

A run varies a declaration in four ways, all decided here from a
:class:`Variant`: ``view`` is typed :class:`~langslice.doors.tools.arguments.ViewAuto`
(with ``resolution``) where the agent chooses the picture size (image
resolution "auto"; the CLI always); ``fit_deformable`` without the image
model has no traced fit sections in its description; with the agent's
``preprocess`` it points there; and with the engine fixed by the user it has
no ``engine`` argument (candidates :class:`~langslice.doors.tools.arguments.FixedCandidate`).

Door-only parameters (ADK's ``tool_context``) are the tool body's, never
declared. Nothing here imports ``google.*``, ``litellm`` or ``openai``.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langslice.core import appearance as looks
from langslice.core.deformation import TRACE_WAIT_S
from langslice.doors.tools.arguments import (
    Candidate,
    DamageEntry,
    FixedCandidate,
    OrientEntry,
    PlacementEntry,
    PositionEntry,
    TransformEntry,
    View,
    ViewAuto,
)

# Defaults of picture tools are read, never mutated (ADK wants a value).
# ruff: noqa: B006

# --- the declarations -------------------------------------------------------------


def status() -> dict[str, Any]:
    """The stack as it stands: one row per section in corrected order.

    Returns:
        ``rows`` (index, id, position_mm, delta_to_next_mm, flip,
        rotation_deg, damaged, damage_note, transform kind, transform_iou,
        transform_mirrored, caveats), plus the stack's cutting angles and
        interval breaks. Writes return only the rows they changed; this is
        the whole table.
    """
    ...


def view_slices(
    slices: list[str],
    view: View = {},
) -> dict[str, Any]:
    """Look at up to 4 named sections at higher resolution.

    Sections are rendered as corrected: any rotation and flip already
    applied, framed to their tissue the same way atlas sections are.
    Each image carries its corrected index and filename burned into its
    top-left corner.

    Args:
        slices: Filenames or corrected indices (max 4 per call).
        view: Picture options (described once in the job statement).
            Modes: "section" (default: the section in view.channels) or
            "channels" (one small tile per raw channel, unmodified, each
            labelled with its name). No atlas is drawn.

    Returns:
        status/slices/description plus the images, in the order asked.
    """
    ...


def view_atlas(
    positions_mm: list[float],
    view: View = {},
) -> dict[str, Any]:
    """Look at atlas sections at the positions you name, at most 4 per call.

    Sections are rendered at the stack's current cutting angles, each
    labelled with its position (and the angles, when the stack is oblique)
    in its top-left corner. Ask for more than 4 and only the first 4 are
    shown; the rest come back under ``dropped_positions_mm`` with
    ``truncated: true``. Positions outside the atlas range are clamped,
    and positions within 0.02 mm of one already in the same call are
    coalesced.

    Args:
        positions_mm: Positions along the slicing axis, in millimetres.
        view: Picture options (described once in the job statement).
            Mode "template" only: the atlas alone, framed to its anatomy;
            atlas_channels default ["ara"], add "borders" for the region
            lines. No section is drawn, so channels does not apply.

    Returns:
        status/positions plus the atlas images, in the order requested.
    """
    ...


def note(text: str) -> dict[str, Any]:
    """Append one line to the run notes, which are saved with the results.

    Args:
        text: The note.
    """
    ...


def undo() -> dict[str, Any]:
    """Undo the last write. One tool call undoes as one step."""
    ...


def redo() -> dict[str, Any]:
    """Redo the write that ``undo`` reversed."""
    ...


def mark_damaged(entries: list[DamageEntry]) -> dict[str, Any]:
    """Set or clear damage flags for sections with unreliable outlines.

    Damage here means the section outline massively deviates from the
    atlas: large missing chunks, a missing hemisphere or olfactory bulb,
    split or independently rotated hemispheres, displaced fragments.
    Bubbles, stains, low contrast and small tears with an intact outline
    are NOT damage.

    Args:
        entries: Objects with id (filename or corrected index), damaged
            (boolean, default True) and note. Set damaged=False to clear
            the flag and its note.

    Returns:
        The rows this call changed.
    """
    ...


def preprocess(
    target: str = "both",
    slices: list[str] = [],
    channel_weights: list[float] = [],
    clahe_clip: float = looks.DEFAULT_CLAHE_CLIP,
    clahe_tiles: int = looks.DEFAULT_CLAHE_TILES,
    n4: bool = False,
    denoise: bool = False,
    reset: bool = False,
    view: View = {},
) -> dict[str, Any]:
    """Set how sections look: for what you view, for what a fit reads, or both.

    Sets the appearance of the whole stack (no slices) or of named
    sections (overriding the stack's), for target "view" (every picture
    you are shown from now on), "fit" (the image a deformable fit reads)
    or "both"; each target keeps its own setting. The image is built from
    the section's raw channels: each channel with weight above zero is
    optionally N4 bias-field corrected and denoised (ANTs), contrast-
    enhanced by CLAHE, then the channels are blended by their weights.
    Until this is called both targets use the default appearance.
    Writes, checkpoints and can be undone; another call replaces the
    setting. Picture options never change it.

    Args:
        target: "view", "fit" or "both".
        slices: Filenames or corrected indices; empty sets the stack.
        channel_weights: One weight per raw channel, in the order the
            result's `channels` lists them; 0 leaves a channel out. Give
            the counterstain that lights all the tissue (DAPI, Nissl) the
            most weight and sparse labels (tracers, reporters) little or
            none. Empty: automatic weights by tissue coverage.
        clahe_clip: CLAHE clip limit, 0 (no CLAHE) to 40; the default
            appearance uses 4.
        clahe_tiles: CLAHE tiles per side, 1 to 32; the default uses 8.
        n4: ANTs N4 bias-field correction of uneven illumination.
        denoise: ANTs denoising.
        reset: True returns the target to the default appearance (named
            sections: back to the stack's setting); other settings are
            ignored.
        view: Picture options (described once in the job statement).
            Mode "section" only; zoom applies. The pictures show the
            target's appearance, so channels and the atlas keys do not
            apply.

    Returns:
        The settings in force per target, the raw channels, and for each
        affected section (up to 4) two labelled pictures: BEFORE (the
        target's appearance before this call) then AFTER (as it is now).
    """
    ...


def reorder_slices(slices: list[str], after: str = "start") -> dict[str, Any]:
    """Place one or more sections together in the requested order.

    Args:
        slices: Nonempty list of unique section filenames. These sections
            move as one block in the listed order; all unlisted sections
            keep their relative order. List the whole stack to set its order.
            Use filenames, never corrected indices: this call changes indices.
        after: Filename the block should follow, or "start" (default) to
            put it first. The anchor must not be in slices.

    Returns:
        Changed rows and moved ids. Only corrected indices change; positions
        and transforms are kept. The whole call is one undoable write.
    """
    ...


def set_positions(
    entries: list[PositionEntry],
    view: View = {},
) -> dict[str, Any]:
    """Write positions for one or more sections, and show each placement.

    Positions are in atlas-native millimetres along the slicing axis. A
    value outside the atlas range is clamped and reported back.

    Args:
        entries: ``[{"id": "<filename>", "position_mm": <number>}]``.
        view: Picture options (described once in the job statement).
            Modes as in `view_placement`; default "stacked".

    Returns:
        What was written, what was clamped, the rows it changed, and one
        picture per placement not already seen in a full-canvas,
        atlas-bearing view with this orientation and these cutting
        angles, labelled in the top-left corner.
    """
    ...


def view_placement(
    entries: list[PlacementEntry],
    view: View = {},
) -> dict[str, Any]:
    """Show sections in their complete current registration, or at candidate positions.

    Writes nothing. At most 4 section-position pairs per call. The
    physical modes draw the section on a millimetre-true canvas under its
    complete current registration: its in-plane transform (identity when
    it has none) and, at its own position, its applied deformation (a
    warped section; view.deformation "none" shows the linear placement
    alone). One image per pair.

    Args:
        entries: ``[{"id": "<filename or corrected index>",
            "positions_mm": [<mm>, ...]}]``. An empty or missing
            ``positions_mm`` means that section's current position.
        view: Picture options (described once in the job statement).
            Modes: "template" (default: the atlas alone on the section's
            own canvas, at its scale; the section is in the opening
            message), "overlay" (the section under its registration with
            the atlas lines on it), "checkerboard" (section and atlas
            image in alternating tiles), "outlines" (atlas lines and the
            section's silhouette on black), "section" (the registered
            section alone), "stacked" (one picture: the section as
            corrected over the atlas, each tissue-framed; no placement
            drawn) or "side_by_side" (separate original section and
            atlas images, tissue-framed, full view only, up to 8
            images). zoom and deformation apply to the physical modes.

    Returns:
        The section-position pairs shown, in order, each with its
        calibration, the transform drawn and whether a deformation was
        drawn, and the images per pair in that order.
    """
    ...


def view_stack(
    view: View = {},
) -> dict[str, Any]:
    """The whole stack ordered by written position, each over its atlas match.

    One contact sheet: every section as a labelled thumbnail, in the
    order of the positions written so far (unplaced sections last), a
    placed section with the atlas section at its position pasted
    directly beneath it. The label carries the corrected index,
    filename, position and the signed distance to the next placed
    section. Then one plot of position against corrected index (damaged
    sections in red). Two images; `view_slices` shows any section large.

    Args:
        view: Picture options (described once in the job statement).
            Mode "stacked" only; atlas_channels default ["ara"]; no zoom.

    Returns:
        The rows in that order and the two images.
    """
    ...


def search_position(id: str, window_mm: float, angles: bool) -> dict[str, Any]:
    """Search the atlas around a section's current position. Writes nothing.

    Scores the section against resampled atlas planes and returns the best
    one it found.

    Args:
        id: Filename or corrected index. The section must already
            have a position.
        window_mm: Half-width of the position search, in millimetres.
        angles: True also searches the cutting angles; False holds them at
            the stack's current ones.

    Returns:
        The best position (and angles) with its score.
    """
    ...


def orient_slices(
    entries: list[OrientEntry],
    view: View = {},
) -> dict[str, Any]:
    """Set the flip and rotation of one or more sections, and show them.

    Orientation is part of in-plane alignment: a flip is the sign of the
    section's affine. Corrections are recorded as data; the user's image
    files are never modified. Rotation is applied first, then the flip. A
    section whose orientation changes loses its transform. Determining
    hemisphere orientation (whether a section is mirrored) is only
    possible when there is a visible notch or a noticeable oblique cutting
    angle that produces differences between the hemispheres' anatomy.

    Args:
        entries: ``[{"id": "<filename>", "flip": true|false,
            "rotate_deg": 0|90|180|270}]``. Either key may be omitted to
            leave that correction as it is.
        view: Picture options (described once in the job statement).
            Mode "section" only: each section as it now stands, in
            view.channels. No atlas is drawn.

    Returns:
        The rows this call changed, and each changed section rendered as
        it now stands (up to 4), its corrected index and filename burned
        into its top-left corner.
    """
    ...


def fit_affine(
    slices: list[str],
    method: str = "elastix",
    fit_atlas: str = "",
    include: list[str] = [],
    exclude: list[str] = [],
    view: View = {},
) -> dict[str, Any]:
    """Fit an in-plane affine per section against its atlas section.

    "elastix" (default) refines the section's current transform (none
    yet: from no transform) by matching the section's fit appearance
    against an atlas image, inner anatomy included; it adjusts from there
    and does not search from scratch. "silhouette" fits the tissue
    outline to the atlas outline from scratch (outlines only). Without
    regions a damaged section is refused. Each fit is written as the
    section's transform (undoable, and `adjust_transforms` overwrites it).

    Args:
        slices: Filenames or corrected indices; empty means every
            positioned, undamaged section.
        method: "elastix" (default) or "silhouette".
        fit_atlas: The atlas image the elastix method matches: "ara" (the
            reference template; default) or "nissl" (a Nissl-stained
            reference, hosts with ABBA's atlas). Not for "silhouette".
        include: Regions (acronyms or ids, descendants included) to fit
            by: only the atlas within 300 um of them, against the tissue
            the fit lays there. With "silhouette" they count only where
            they reach the outline.
        exclude: Regions removed from the atlas side (e.g. tissue missing
            from the section), descendants included; the tissue the fit
            lays on them is left out too. With regions given, damaged
            sections are fitted. An include or exclude entry may name one
            side only, "CTX:left" or "CTX:right": left and right of the
            section as view_slices shows it.
        view: Picture options (described once in the job statement).
            Modes as in `adjust_transforms` without "ab"; default
            "overlay". Included regions are highlighted unless
            view.regions names others.

    Returns:
        Per-section overlap (iou; with regions, of the kept atlas and the
        tissue that corresponds), the transform as the five physical
        knobs about the canvas centre, the calibration the image was
        drawn with, a `regions` report when regions were given, and an
        image of each fitted section under its new transform. The
        generic changed row is omitted because it repeats the same fit
        identifiers and overlap.
    """
    ...


def adjust_transforms(
    entries: list[TransformEntry],
    view: View = {},
) -> dict[str, Any]:
    """Set and show one to four independent sections in one undoable call.

    Each entry replaces the complete transform; a shear left out is kept from
    the current transform. A section may appear once per call; inspect its
    result before making a dependent correction in a later call. Call it as
    often as you need, on any section that has a position; the last call is
    what stays.

    Args:
        entries: One to four objects with id, rotation_deg (counter-clockwise
            about the pivot, degrees), scale_x, scale_y (multipliers about
            the pivot; 1.0 leaves the size alone), translate_x_mm (right),
            translate_y_mm (down). Optional per entry: shear (a unitless
            slant applied before the rotation: each point moves sideways
            by shear times its distance below the pivot, in units of
            scale_x; the number fit_affine reports; left out, the
            section's current shear is kept, 0 sets none), pivot
            ("canvas", "tissue" or [fx, fy] fractions of the canvas) and
            note.
        view: Picture options (described once in the job statement), one
            for every entry's picture. Modes: "overlay" (default: the section
            with the atlas lines on it), "side_by_side" (two images: the
            section, then the atlas image, same scale and crop),
            "checkerboard", "outlines" (the atlas lines and the section's
            own silhouette on black), "section", "template" (the atlas
            alone) or "ab" (two overlays at the same crop: these
            parameters, then the section's stored transform, or identity
            when it has none).

    Returns:
        Per-section results with zero-based image_indexes into the attached
        labelled images, in entry order, and the `view` they were drawn
        with. All writes form one undo step. Repeating unchanged
        parameters only redraws, without an undo step.
    """
    ...


def set_cutting_angles(pitch_deg: float, yaw_deg: float) -> dict[str, Any]:
    """Set the stack-wide cutting angles.

    Subsequent atlas fetches and previews are rendered at these angles.

    Args:
        pitch_deg: Rotation about the plane's column axis, in degrees.
        yaw_deg: Rotation about the plane's row axis, in degrees.

    Returns:
        The rows this call changed.
    """
    ...


def trace_borders(id: str, prompt: str = "") -> dict[str, Any]:
    """Trace one slice's atlas borders onto its anatomy with the image model.

    Args:
        id: Section filename or corrected index, with a position and linear transform.
        prompt: The full image prompt for this section, edited from the base prompt.

    Starts the image call in the background and returns at once; the result is
    saved, and submit waits for it. The first result at a placement is reused.
    Does not fit a deformation.
    """
    ...


def trace_from_atlas(slices: list[str], passes: int = 1) -> dict[str, Any]:
    """Trace sections' atlas borders with the image model, shown no placement.

    The model sees each clean section and the outlined atlas plane at its
    position and cutting angles; passes=2 adds a corrective second call. The
    reply is recorded as trace_borders records its own, so fit_deformable's
    traced fit sections start it from the section's linear transform.

    Args:
        slices: Filenames or corrected indices, each with a position and
            linear transform.
        passes: 1, or 2 for the corrective second call.

    Starts the image calls in the background and returns at once; the
    results are saved, and submit waits for them. Does not fit a deformation.
    """
    ...


def grep_atlas(query: str, section: str = "") -> dict[str, Any]:
    """Look regions up in the atlas hierarchy, like grepping the ontology.

    Args:
        query: Text matched case-insensitively against region acronyms and
            names (substring), or an exact acronym or numeric id.
        section: Optional filename or corrected index of a section with a
            position; each row then says whether the region (or any
            descendant) appears in the atlas plane at that placement.

    Returns:
        Rows of acronym, id, name, ancestry (root to parent, as acronyms)
        and descendant count, capped at 40 with the number left over.
    """
    ...


def fit_deformable(
    slices: list[str],
    include: list[str] = [],
    exclude: list[str] = [],
    start: str = "linear",
    fit_section: str = "fit",
    fit_atlas: str = "",
    engine: str = "",
    stiffness: str = "medium",
    candidates: list[Candidate] = [],
    keep_linear: str = "",
    view: View = {},
) -> dict[str, Any]:
    """Fit a deformation of the placed atlas onto sections, on top of their linear placement.

    A library engine bends the atlas, as linearly placed, onto the
    section image. A call with several candidates previews them all
    (run concurrently) and writes nothing. A call with exactly one
    setting (no candidates, or one) APPLIES it as each section's
    deformation, reusing the result of an identical earlier fit instead
    of recomputing; that write is undoable and checkpointed. Any later
    change to a section's position, orientation, cutting angles or
    transform clears its deformation. Needs a position and a transform.
    With keep_linear, no fit runs: each named section records that its
    linear placement stands, with that reason.

    Args:
        slices: Filenames or corrected indices (up to 4; at most 8 fits
            per call, slices times candidates).
        include: Regions (acronyms or ids, descendants included) to focus
            on: only they and a 300 um margin are fitted. Empty fits the
            whole section.
        exclude: Regions removed from the atlas side (e.g. tissue that is
            missing from the section), descendants included. An include
            or exclude entry may name one side only, "CTX:left" or
            "CTX:right": left and right of the section as this tool's
            pictures show it.
        start: "linear" (from the linear placement) or "current" (compose
            onto the section's applied deformation: region-by-region steps).
        fit_section: What of the section the fit reads: "fit" (the
            section's fit appearance; default), "traced_borders" (the
            section's trace_borders result at this placement, its lines
            turned into named regions; ANTs) or "traced_lines" (those
            lines as lines, against atlas borders). A traced fit section
            waits for a trace still running (up to TRACE_WAIT minutes)
            and the reply adds the trace drawn on the section. With a
            completed trace, traced_borders with the ANTs engine at medium
            stiffness is the recommended pairing.
        fit_atlas: What of the atlas the fit reads. For "fit": "ara" (the
            atlas's reference template; default) or "nissl" (a
            Nissl-stained reference, hosts with ABBA's atlas). For traced
            fit sections: "borders" (default).
        engine: "ants" or "elastix"; empty is ANTs when installed.
        stiffness: "soft", "medium" (default) or "firm".
        candidates: 2 to 4 objects, each overriding any of stiffness,
            fit_section, fit_atlas and engine for one variant.
        keep_linear: A reason the named sections' linear placement stands
            without a deformation. Given, nothing is fitted and nothing is
            drawn: each section records it at its current placement (one
            undo step; submit accepts it like an applied fit; a placement
            change clears it).
        view: Picture options (described once in the job statement).
            Modes: "borders" (default: the fitted borders on the image the
            fit read) or "ab" (that, then what the fit started from).
            atlas_channels default ["borders"]; add "ara" or "nissl" to
            see that atlas image, warped, under the lines at
            atlas_opacity. view.regions is drawn at full strength (empty:
            the include list); excluded regions are drawn in pink. The
            pictures are the image the fit read, so channels does not
            apply.

    Returns:
        Per section and candidate: the settings and engine numbers used,
        displacement (max and median, mm, over the tissue), fold fraction,
        plausibility flags (regions compressed, expanded, vanished or
        folded beyond limits; displacement outsized for the section) and
        image_indexes into the pictures: the final borders drawn on the
        section image the fit read. Traced sections add `traces`: each
        one's trace drawn on the section.
    """
    ...


def submit(
    summary: str,
    notes: list[str],
    interval_breaks: list[int],
) -> dict[str, Any]:
    """End the run. Call this exactly once, last.

    Args:
        summary: One or two sentences on what you did.
        interval_breaks: Corrected indices of the sections AFTER a gap you
            conclude is real. Empty if there are none.
        notes: Short observations worth carrying forward.
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
        slices: Filenames or corrected indices; empty is every section.
        full_resolution: Write the maps on the image file's own pixels
            instead of its working copy (what every picture is drawn from).

    Returns:
        written (the sections), skipped (each with its reason), files
        (path and kind), seconds.
    """
    ...


#: Every declared verb, in :data:`langslice.ops.registry.VERBS` order.
STUBS: dict[str, Callable[..., Any]] = {
    stub.__name__: stub for stub in (
        status, view_slices, view_atlas, note, undo, redo, mark_damaged, preprocess,
        reorder_slices, set_positions, view_placement, view_stack,
        search_position, orient_slices, fit_affine, adjust_transforms, set_cutting_angles,
        trace_borders, trace_from_atlas, grep_atlas, fit_deformable, submit, export_maps,
    )
}

# --- the run's variant ------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """What of a run changes its declarations (see the module text)."""

    #: The image model is in the run (``trace_borders``, traced fit sections).
    traces: bool = True
    #: The agent may set appearances (``preprocess``).
    preprocessing: bool = False
    #: The deformable engine: "either" (the agent chooses) or the user's fixed one.
    engine: str = "either"
    #: The caller chooses each picture's size (``view.resolution``).
    auto: bool = False
    #: Who reads the declaration (:data:`DOORS`): where the opening pictures
    #: are, and how an image-model verb answers.
    door: str = "agent"

    @classmethod
    def of(cls, spec: Any, *, auto: bool, image_model: bool = True,
           door: str = "agent") -> Variant:
        """The variant of a :class:`~langslice.core.spec.JobSpec`'s run;
        *image_model* False: the door cannot reach its image model, so the
        run is declared as one without it."""
        return cls(traces=bool(spec.nonlinear.uses_image_model and image_model),
                   preprocessing=bool(spec.agent_preprocessing),
                   engine=str(spec.nonlinear.engine), auto=bool(auto), door=door)


#: The doors a verb is declared for: ``agent`` (LangSlice's own agent, and
#: the library), ``mcp`` (Claude Desktop) and ``cli`` (the agent CLI).
DOORS = ("agent", "mcp", "cli")

#: Passages each door words its own way, as ``(agent text, {door: text})``
#: per verb: where the opening pictures are (the ADK agent's seed message,
#: the MCP door's ``show_stack`` pages, the CLI's ``brief`` files), and how
#: an image-model verb answers (the CLI waits for the call to land unless
#: run with ``--background``).
_DOOR_DOCS: dict[str, tuple[tuple[str, dict[str, str]], ...]] = {
    "view_placement": ((
        "the section is in the opening\n                message)",
        {"mcp": "the section is in the opening\n                pictures, show_stack)",
         "cli": "the section is in the opening\n                pictures, brief)"},
    ),),
    "trace_borders": ((
        "Starts the image call in the background and returns at once; the result is\n"
        "        saved, and submit waits for it.",
        {"cli": "Runs the image call and answers once it has landed (with --background\n"
                "        at once; `wait` collects the answer); the result is saved, and\n"
                "        submit waits for it."},
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


#: The variant a caller without a job sees: every argument, every option.
#: The CLI's ``langslice schema`` uses it with ``auto`` (the CLI's pictures
#: are sized by the caller).
FULL = Variant()

# --- fit_deformable's description per run ------------------------------------------

#: The ``fit_deformable`` description's recommendation for traced fit sections
#: (a small correction of a good placement); dropped where it cannot apply (no
#: image model, or the user fixed the engine to Elastix).
_RECOMMENDED_TRACED = (
    " With a\n"
    "                completed trace, traced_borders with the ANTs engine at medium\n"
    "                stiffness is the recommended pairing."
)
#: The ``fit_deformable`` description's line on the fit appearance, and the
#: pointer to ``preprocess`` added when the agent may set appearances.
_FIT_LOOK_DOC = (
    '            fit_section: What of the section the fit reads: "fit" (the\n'
    "                section's fit appearance; default"
)
_PREPROCESS_DOC = (
    ";\n                the preprocess tool, target \"fit\", sets it, e.g. to one raw channel"
)
#: ``fit_deformable`` passages about traced fit sections and their wording for
#: a run without the image model (``nonlinear.provider`` "none").
_STAIN_ONLY_DOC: tuple[tuple[str, str], ...] = (
    (
        _FIT_LOOK_DOC + '), "traced_borders" (the\n'
        "                section's trace_borders result at this placement, its lines\n"
        '                turned into named regions; ANTs) or "traced_lines" (those\n'
        "                lines as lines, against atlas borders). A traced fit section\n"
        "                waits for a trace still running (up to TRACE_WAIT minutes)\n"
        "                and the reply adds the trace drawn on the section."
        + _RECOMMENDED_TRACED + "\n",
        _FIT_LOOK_DOC + ").\n",
    ),
    (
        '            fit_atlas: What of the atlas the fit reads. For "fit": "ara" (the\n'
        "                atlas's reference template; default) or \"nissl\" (a\n"
        "                Nissl-stained reference, hosts with ABBA's atlas). For traced\n"
        '                fit sections: "borders" (default).\n',
        '            fit_atlas: What of the atlas the fit reads: "ara" (the atlas\'s\n'
        '                reference template; default) or "nissl" (a Nissl-stained\n'
        "                reference, hosts with ABBA's atlas).\n",
    ),
    (" Traced sections add `traces`: each\n            one's trace drawn on the section.", ""),
)


def _fit_deformable_doc(doc: str, variant: Variant) -> str:
    wait = f"{TRACE_WAIT_S / 60:g} minutes"
    doc = doc.replace("TRACE_WAIT minutes", wait)
    if not variant.traces:
        for traced_text, stain_text in _STAIN_ONLY_DOC:
            doc = doc.replace(traced_text.replace("TRACE_WAIT minutes", wait), stain_text)
    if variant.preprocessing:
        doc = doc.replace(_FIT_LOOK_DOC, _FIT_LOOK_DOC + _PREPROCESS_DOC)
    if variant.engine == "either":
        return doc
    # The user fixed the engine: the same tool without the engine argument.
    doc = doc.replace(
        '            engine: "ants" or "elastix"; empty is ANTs when installed.\n', "",
    ).replace("A library engine", f"The {variant.engine} engine").replace(
        "fit_section, fit_atlas and engine for one variant", "fit_section and fit_atlas for one "
        "variant")
    if variant.engine != "ants":
        # traced_borders is ANTs-only, so its recommendation does not apply.
        doc = doc.replace(_RECOMMENDED_TRACED, "")
    return doc


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


@functools.cache
def declaration(name: str, variant: Variant = FULL) -> Declaration:
    """The declaration of verb *name* in a run of *variant*."""
    stub = STUBS[name]
    signature = inspect.signature(stub, eval_str=True)
    annotations = dict(inspect.get_annotations(stub, eval_str=True))
    parameters = list(signature.parameters.values())
    doc = _door_doc(name, model_doc(stub), variant.door)
    if name == "fit_deformable":
        doc = _fit_deformable_doc(doc, variant)
        if variant.engine != "either":
            parameters = [
                p.replace(annotation=list[FixedCandidate]) if p.name == "candidates" else p
                for p in parameters if p.name != "engine"
            ]
            annotations.pop("engine", None)
            annotations["candidates"] = list[FixedCandidate]
    if variant.auto and "view" in annotations:
        parameters = [p.replace(annotation=ViewAuto) if p.name == "view" else p
                      for p in parameters]
        annotations["view"] = ViewAuto
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
    is not declared is door-only and keeps *body*'s default, except ADK's
    ``tool_context``, which is added to the signature so ADK passes it.
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
