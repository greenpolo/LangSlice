"""The changing tools' descriptions: positions, transforms, damage and masks.

Each tool's description is data, read as
:mod:`~langslice.tools.descriptions.description` says.
"""

from __future__ import annotations

from typing import Literal

from langslice.tools.arguments import (
    CuttingAngles,
    SectionPosition,
    SectionTransform,
)
from langslice.tools.descriptions.description import (
    REQUIRED,
    Argument,
    Offered,
    PerCap,
    ToolDescription,
)

#: Where ``make_mask`` takes its sorting from.
MaskSource = Literal["em", "fit"]

POSITION_SECTIONS = ToolDescription(
    summary=(
        "Set where sections sit along the slicing axis, and the stack's "
        "cutting angles."
    ),
    body=(
        """\
Positions are atlas millimetres along the slicing axis; a value outside
the atlas range is clamped into it and reported. The stack's order
follows the positions. A changed position or angle clears the
deformation of each section it moves (the reply lists them). One
undoable write.""",
        """\
Cutting angles tilt the atlas plane every section is cut at; every
section gets the same angles, and every later atlas picture is drawn at
them. Pitch 0 and yaw 0 is the atlas's flat plane. A tilt moves each
edge of the plane along the slicing axis: an edge D mm from the
picture's centre moves about D x tan(angle) mm (10 degrees moves an edge
4 mm from the centre about 0.7 mm). Which edge moves which way, for
positive angles, by the plane the stack is cut in (edges as the pictures
show them):""",
        """\
- coronal: pitch puts the top edge at a larger position (mm) than the
  bottom edge; yaw puts the right edge at a larger position than the
  left edge.
- sagittal: pitch puts the left edge at a larger position than the
  right edge; yaw puts the bottom edge at a larger position than the
  top edge.
- horizontal: pitch puts the right edge at a larger position than the
  left edge; yaw puts the top edge at a larger position than the
  bottom edge.""",
        "Negative angles move the edges the opposite way.",
    ),
    arguments={
        "sections": Argument(
            list[SectionPosition],
            [],
            """\
[{"id": "<filename>", "position_mm": <number>}]; the
extension may be left off.""",
            offered=Offered.POSITIONING,
        ),
        "cutting_angles": Argument(
            CuttingAngles,
            {},
            """\
{"pitch_deg": <number>, "yaw_deg": <number>} for the
whole stack; empty leaves the angles as they are.""",
        ),
        "view": Argument(
            bool,
            True,
            "True returns the picture described below; false returns none.",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
What was written and what was clamped, the stack's order after the
call, the rows that changed, and [[?look:positioning a positioning picture of \
the
written sections: the atlas at their new positions above a
millimetre ruler, the sections below it, each joined by a line to
its position.||a picture of each written section with the atlas
borders of its new position on it.]]""",
)

INTERACTIVE_TRANSFORM = ToolDescription(
    summary="""\
Set sections' [[?interactive_transform:flip|interactive_transform:rotate \
orientation and ]]in-plane transform by hand, and show the result.""",
    body=(
        """\
For a few sections at a time, in a look-and-adjust loop. Values are
absolute: a value given replaces the section's current one, and a value
left out keeps it (a section without a transform starts from the
identity). [[?interactive_transform:rotate The quarter turn is applied first\
[[?interactive_transform:flip , then the flip]]. ]]\
[[?interactive_transform:flip|interactive_transform:rotate A changed
orientation keeps the other values: the transform is rebuilt on the new
orientation, and the reply says so. ]]One undoable write; the image files
are never modified.""",
        """\
[[?interactive_transform:flip \
Whether a section is mirrored can only be decided from a visible notch,
or from a cutting angle oblique enough that the two hemispheres'
anatomy differs.]]""",
    ),
    arguments={
        "sections": Argument(
            list[SectionTransform],
            REQUIRED,
            """\
${count} objects: "id" (filename; the extension may be
left off) and any of [[?interactive_transform:flip "flip" (true mirrors the \
section
left-right), ]]\
[[?interactive_transform:rotate "rotate_quarter" (0, 90, 180 or 270 degrees
counter-clockwise), ]]"rotation_deg" (degrees counter-clockwise
about the pivot, the canvas centre),
"scale_x" and "scale_y" (multipliers about the pivot; 1.0 keeps
the size), "shear" (a slant applied before the rotation: each
point moves sideways by shear times its distance below the pivot,
in units of scale_x), "translate_x_mm" (right) and
"translate_y_mm" (down). A section appears once per call.""",
        ),
        "view": Argument(
            bool,
            True,
            "True returns the picture described below; false returns none.",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
Per section [[?interactive_transform:flip|interactive_transform:rotate \
its orientation, ]]its transform as the values above,
and whether anything was written; and a picture of each section
under its new transform with the atlas borders on it.""",
    slots={"count": PerCap(capped="One to four", uncapped="One or more")},
)

MARK_DAMAGE = ToolDescription(
    summary=(
        "Mark the atlas regions a section has lost, so that every fit leaves "
        "them out."
    ),
    body=(
        """\
Mark the regions whose tissue is missing from the section, or lies folded
over other anatomy, such as a whole hemisphere, the olfactory bulb or the
cortex: the fits leave them out, and the anatomy around them fits
cleanly. Tissue that is intact but shifted near its place is not damage;
left unmarked, the warp fits it.[[?landmark_triangulation  Tissue shifted \
further is not
damage either: move it with landmarks.]] A marked region is never fitted, so
intact tissue inside a mark goes unfitted too. When most of a large
region is missing, marking the whole region keeps the anatomy around it
safe. Small tears, bubbles, stains, low contrast and folds within an
intact outline are not damage and are not marked. A section is damaged
exactly when it has marked regions. \
[[?elastix_affine|fireants_syn|trace_borders The marked
regions are left out of \
[[#elastix_affine=elastix_affine||fireants_syn=fireants_syn||\
trace_borders=the image model's
trace]] automatically, and they||The marked regions]] move with the \
registration because they
are named by atlas region. One undoable write.""",
    ),
    arguments={
        "section": Argument(
            str,
            REQUIRED,
            "Filename (the extension may be left off).",
        ),
        "regions": Argument(
            list[str],
            REQUIRED,
            """\
Acronyms, names or ids, descendants included; "CTX:left" or
"CTX:right" names one side, as the pictures show the section.
They replace the section's marked regions; empty clears them.""",
        ),
        "note": Argument(
            str,
            "",
            """\
What is wrong with the tissue. A note the user gave stays
first.""",
        ),
    },
    returns="""\
The section's marked regions and note, and a picture of the marked
regions shaded on the section under its current registration beside
the same regions on the atlas, to check both at once.""",
)

MAKE_MASK = ToolDescription(
    summary=(
        "Make, replace or delete a named mask of sections: which tissue a fit "
        "compares."
    ),
    body=(
        """\
A mask holds a weight from 0 to 1 for each pixel of a section; a fit
that takes it by name ignores the pixels under 0.5. It sorts the
section's pixels into tissue, artifact and background by their brightness
in a section image and by where the atlas puts tissue, as the fits' own
estimate does, and it is the only way a fit leaves out a hole, a tear or a
stain. Each section gets its own mask, made at its own current
registration. The mask sits on the section's own pixels, so it stays usable \
after
fits and moves; status says when it was made at an earlier registration.
Turning or flipping the section makes a fit refuse it until it is made
again. The marked damage regions are not part of a mask; every
fit leaves them out of the atlas on its own. Undoable.""",
    ),
    arguments={
        "name": Argument(
            str,
            REQUIRED,
            "The mask's name; an existing name is replaced.",
        ),
        "sections": Argument(
            list[str],
            [],
            """\
Filenames (the extension may be left off) of the sections to
make it for; empty is every section.""",
        ),
        "source": Argument(
            MaskSource,
            "em",
            """\
"em": sort each section's pixels now, at its current
registration, against atlas_image. "fit": the sorting the
section's applied deformation saved with it, copied as it was
(NO_DEFORMATION or NO_SORTING when it has none).""",
        ),
        "section_image": Argument(
            str,
            "preprocessed",
            """\
The section image to sort ("preprocessed", a raw channel\
[[?make_image  or an image made with make_image]]); "em" only.""",
        ),
        "atlas_image": Argument(
            str,
            "nissl",
            """\
The atlas image the sorting compares with ("nissl" is the
atlas's Nissl image where it has one, else its template); "em"
only.""",
        ),
        "sorting": Argument(
            dict[str, float],
            {},
            """\
Starting points of the sort, each between 0 and 1 of the
section image's brightness or a spread in the same units: match_sd,
background_sd, artifact_sd, background_mean, artifact_mean. Empty
uses the defaults; "em" only.""",
        ),
        "delete": Argument(
            bool,
            False,
            """\
True removes the mask (from the named sections, or
everywhere); the other arguments except name are then ignored.""",
        ),
        "view": Argument(
            bool,
            True,
            "True returns the pictures described below; false returns none.",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
Per section the mask's source, section image, atlas image and
starting points, and the fraction of the section's tissue it leaves
out, or why the section has no mask. With view, ${sections_shown}, each \
drawn as
look's "section" mode draws the image it sorted, with the mask's 0.5 outline \
on it,
and a caption saying the outlines trace the mask's edge, that the
ringed patches are what the mask leaves out of a fit, and the mask's
name and source.""",
    slots={
        "sections_shown": PerCap(
            capped="""\
up to 4 of the
sections (spread over the stack when none are named)""",
            uncapped="""\
each of the sections
(the whole stack when none are named)""",
        ),
    },
)

DESCRIPTIONS: dict[str, ToolDescription] = {
    "position_sections": POSITION_SECTIONS,
    "interactive_transform": INTERACTIVE_TRANSFORM,
    "mark_damage": MARK_DAMAGE,
    "make_mask": MAKE_MASK,
}
