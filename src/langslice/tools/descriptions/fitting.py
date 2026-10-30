"""The fitting tools' descriptions: elastix, FireANTs SyN and the border trace.

Each tool's description is data, read as
:mod:`~langslice.tools.descriptions.description` says.
"""

from __future__ import annotations

from typing import Literal

from langslice.core.fits.deformable.settings import DEFAULT_NEIGHBOURHOOD_UM
from langslice.tools.descriptions.description import (
    REQUIRED,
    Argument,
    Distance,
    Offered,
    PerCap,
    PerDoor,
    ToolDescription,
)
from langslice.tools.descriptions.landmarks import LANDMARK_MOVE_UM

#: How ``fireants_syn`` compares the two images.
FitMetric = Literal["mutual_information", "local_correlation"]
#: The resolution ``fireants_syn`` works at.
FitDetail = Literal["coarse", "standard", "fine"]

ELASTIX_AFFINE = ToolDescription(
    summary=(
        "Fit sections' in-plane affine transforms automatically with elastix."
    ),
    body=(
        """\
For a few sections at a time. Elastix refines each section's current
placement (from the identity when it has no transform) by matching a
section image against an atlas image, inner anatomy included. It
adjusts from where the
section is and does not search from scratch: it moves a section by
small amounts and does not turn it over. The section's marked damage
regions are left out of the atlas automatically. A section that holds a
deformation is refused with HAS_DEFORMATION: undo the deformation
first[[?fireants_syn , or fit a warp on it]]. Each fit is written as the
section's transform, one undoable write for the call.""",
        """\
Use restrict_to when a few clearly recognizable, surviving regions
provide better registration evidence than the whole section. Those
regions guide a whole-section affine transform. Review the wider
overlay afterward as well.""",
        """\
A region too small for elastix is skipped with REGION_TOO_SMALL; use
a larger region[[?interactive_transform  or interactive_transform]]. No fit \
is written for it.""",
        """\
Both images are scaled from their 1st to 99th percentile inside the
atlas outline to 0-1. The reply names the section image, atlas image and mask
each fit used.""",
    ),
    arguments={
        "sections": Argument(
            list[str],
            REQUIRED,
            "${count} filenames (the extension may be left off).",
        ),
        "restrict_to": Argument(
            list[str],
            [],
            """\
Regions to fit by (acronyms, names or ids, descendants
included; "CTX:left" or "CTX:right" for one side): only the
atlas within ${neighbourhood} of them, against the tissue the fit lays
there. Only the ANTs and Elastix fits take it. Empty fits by
every region.""",
        ),
        "section_image": Argument(
            str,
            "preprocessed",
            """\
The section image to match, by name: "preprocessed", a raw channel\
[[?make_image , or an image made with make_image]].""",
        ),
        "atlas_image": Argument(
            str,
            "nissl",
            """\
One of the atlas's images, by name (the job statement lists
them and says what each is), or "nissl", the atlas's Nissl image
where it has one, else its template.""",
        ),
        "mask": Argument(
            str,
            "none",
            """\
"none" compares every pixel of the section[[?make_mask , or the
name of a mask of that section made with make_mask: pixels with
a weight under 0.5 are left out]].""",
            needs="make_mask",
        ),
        "edges": Argument(
            bool,
            True,
            """\
True also matches the edges (the gradient magnitude) of both
images, as a second term of equal weight.""",
        ),
        "view": Argument(
            bool,
            True,
            "True returns the picture described below; false returns none.",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
Per section the transform (rotation_deg, scale_x,
scale_y, shear, translate_x_mm, translate_y_mm about the canvas
centre); the `inputs` it used (section image, atlas image as
resolved, mask, damage regions left out, intensity scaling); and a
picture of each fitted section
under its new transform with the atlas borders on it. With
restrict_to, those regions' borders are drawn thick and the others
faint, and the picture is zoomed to them when they cover a small
part of the section.""",
    slots={
        "count": PerCap(capped="One to four", uncapped="One or more"),
        "neighbourhood": Distance(DEFAULT_NEIGHBOURHOOD_UM, "{} um"),
    },
)

FIREANTS_SYN = ToolDescription(
    summary=(
        "Fit the atlas onto sections with FireANTs SyN, on their current "
        "registration."
    ),
    body=(
        """\
This is ANTs SyN (symmetric normalization; Avants et al. 2008, Medical Image
Analysis), the most widely used diffeomorphic registration in neuroimaging,
run by FireANTs (Jena et al. 2026, Nature Communications), which keeps SyN's
model and runs its optimisation on the GPU.""",
        """\
SyN bends the atlas onto a section image of each section, starting
from the section's current registration (its linear placement, and its
deformation when it has one), so each fit builds on the one before;
undo goes back. `restrict_to` refines single regions on top of the
current fit.[[?landmark_triangulation  SyN does not move a piece displaced \
more than ${landmark_move} from its place; landmarks do.]] The section's \
marked damage regions are left out of
the atlas automatically. Needs a position and a transform. Applied as the
section's deformation, one undoable write; a later change to the
section's position, orientation, cutting angles or transform clears it.""",
        """\
Use restrict_to when a few clearly recognizable, surviving regions
provide better registration evidence than the whole section, to refine
the existing registration locally. Review the wider overlay afterward
as well.""",
        """\
Both images are scaled from their 1st to 99th percentile inside the
atlas outline to 0-1. The step size and the number of iterations at each
scale are fixed. The reply names the section image, atlas image and mask each
fit used.""",
    ),
    arguments={
        "sections": Argument(
            list[str],
            REQUIRED,
            """\
One to twenty filenames (the extension may be left off); the
fits run one after another.""",
        ),
        "restrict_to": Argument(
            list[str],
            [],
            """\
Regions to fit by (acronyms, names or ids, descendants
included; "CTX:left" or "CTX:right" for one side): the atlas and
the section are both cut to the regions' neighbourhood and the
displacement fades to nothing beyond it. Only the ANTs and
Elastix fits take it. Empty fits by every region.""",
        ),
        "section_image": Argument(
            str,
            "preprocessed",
            """\
The section image to match, by name: "preprocessed", a raw channel\
[[?make_image , or an image made with make_image]].""",
        ),
        "atlas_image": Argument(
            str,
            "nissl",
            """\
One of the atlas's images, by name (the job statement lists
them and says what each is), or "nissl", the atlas's Nissl image
where it has one, else its template.""",
        ),
        "mask": Argument(
            str,
            "none",
            """\
"none" compares every pixel of the section[[?make_mask , or the
name of a mask of that section made with make_mask: pixels with
a weight under 0.5 are left out]].""",
            needs="make_mask",
        ),
        "smoothness_um": Argument(
            float,
            80.0,
            """\
The warp's smoothness scale in micrometres, as for an
adult mouse brain (a larger or smaller brain scales it): the
width over which each update of the displacement field is
smoothed, the whole field a quarter of that. Larger bends less.""",
        ),
        "metric": Argument(
            FitMetric,
            "mutual_information",
            """\
"mutual_information" or "local_correlation": how the two
images are compared.""",
            needs="fireants_syn:metric",
        ),
        "edges": Argument(
            bool,
            True,
            """\
True also matches the edges (the gradient magnitude) of both
images, as a second term of equal weight.""",
            needs="fireants_syn:edges",
        ),
        "detail": Argument(
            FitDetail,
            "standard",
            """\
"coarse", "standard" or "fine": the resolution the fit works
at (fine: 10 um for an adult mouse brain) and the iterations it
takes; finer is slower.""",
            needs="fireants_syn:detail",
        ),
        "view": Argument(
            bool,
            True,
            "True returns the picture described below; false returns none.",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
Per section the displacement (median and max, mm, over the placed atlas),
the fold fraction and plausibility flags (regions compressed,
expanded, vanished or folded beyond limits; displacement outsized
for the section[[?landmark_triangulation , measured beyond any landmark \
warp]]); the
`inputs` it used (section image, atlas image as resolved, mask, damage
regions left out, intensity scaling) and `estimated` (the fraction of
the section the fit left out on its own); and a
picture of each section under its new registration with the atlas borders on \
it. With
restrict_to, those regions' borders are drawn thick and the others faint, and \
the
picture is zoomed to them when they cover a small part of the
section.""",
    slots={
        "landmark_move": Distance(
            LANDMARK_MOVE_UM, "about {} um", mouse_words="about\n400 um"
        ),
    },
)

TRACE_BORDERS = ToolDescription(
    summary=(
        "Trace the atlas borders onto a section with the image model, then "
        "fit it with FireANTs."
    ),
    body=(
        """\
LangSlice's own nonlinear method, packaged: the image model is shown the
section's "preprocessed" image with its placed atlas borders and draws
them on the section's anatomy; when it answers, FireANTs fits the traced \
borders (80 um
smoothness) on top of the section's current registration and the
deformation is applied as its own undoable step. The section needs a
position and a transform; its marked damage regions are left out of
what the model is shown. With restrict_to, the model is shown only those
regions' borders and only they are fitted. The first answer at a
placement and region choice is saved and reused, whatever the prompt;
when its fit is already applied, nothing is fitted again. Every prompt
sent is saved in the job folder.""",
        """\
A focused trace can address one difficult region without tracing the
whole section. Use restrict_to to select it and, when helpful, tailor
the full image prompt to that section while preserving the stated
border-drawing rules. The tool handles both tracing and fitting the result.""",
        """\
${background} the fit's numbers, then its
pictures (the fitted borders, and the trace on the section), each with
its number and caption. status
lists the work still running; submit waits for it. A write to this
section is refused until the work lands.""",
    ),
    arguments={
        "section": Argument(
            str,
            REQUIRED,
            "Filename (the extension may be left off).",
        ),
        "prompt": Argument(
            str,
            "",
            """\
The full image prompt for this section: the base prompt
below, edited for this section; empty sends the base prompt.
Rules for any edit: the atlas borders shown to the model are the
only source of which lines exist, so its answer holds each of
them and no other line, except where tissue is physically missing
from the section; a faint or indistinct boundary is still drawn,
where the atlas places it. Never add a sentence that makes a
line depend on whether its edge is visible.""",
        ),
        "restrict_to": Argument(
            list[str],
            [],
            """\
Regions to trace and fit alone (acronyms, names or ids,
descendants included; "CTX:left" or "CTX:right" for one side).
Empty: every region.""",
        ),
    },
    prompt_heading="""\
The base prompt (its image numbers are the images the image model
receives):""",
    slots={
        "background": PerDoor(
            agent="""\
Starts the work in the background and returns at once with its id, so
you can carry on with other sections. When the work finishes, the next
tool reply starts with its notice:""",
            cli="""\
The command answers once the work has finished (with --background
at once, and `wait` collects the answer), with its notice:""",
        ),
    },
)

DESCRIPTIONS: dict[str, ToolDescription] = {
    "elastix_affine": ELASTIX_AFFINE,
    "fireants_syn": FIREANTS_SYN,
    "trace_borders": TRACE_BORDERS,
}
