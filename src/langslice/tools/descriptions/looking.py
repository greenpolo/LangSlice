"""The looking tools' descriptions: pictures, the atlas, the status, files.

Each tool's description is data, read as
:mod:`~langslice.tools.descriptions.description` says.
"""

from __future__ import annotations

from typing import Literal

from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.sections import images
from langslice.tools.arguments import PicturePoint
from langslice.tools.descriptions.description import (
    REQUIRED,
    Argument,
    Offered,
    PerCap,
    ToolDescription,
)

#: The pictures ``look`` draws, offered as an enum in every door's schema.
LookMode = Literal["section", "atlas", "overlay", "positioning"]
#: Whether ``look`` draws each section with its deformation.
Warp = Literal["applied", "none"]

LOOK = ToolDescription(
    summary="Draw pictures of sections and the atlas. Writes nothing.",
    body=(
        """\
The same request draws the same kind of picture at every stage of the
run: no default depends on the stack's state. Each picture gets a number
and a caption stating the positions, cutting angles, scale and section
image it was drawn with[[?zoom ; zoom takes the number]].""",
        """\
[[?look:positioning Use positioning mode to compare sections with candidate \
atlas planes
along the slicing axis. ]][[?look:overlay Use overlays to assess in-plane \
alignment[[?fireants_syn|landmark_triangulation|trace_borders  and
deformation]] against internal anatomy.]]""",
    ),
    arguments={
        "mode": Argument(
            LookMode,
            REQUIRED,
            """\
[[?look:section "section": each section alone, as oriented, the whole
image. ]][[?look:atlas "atlas": the atlas plane alone at each of positions_mm,
at the stack's cutting angles; or, for each section named, at that
section's own position and cutting angles (positions_mm then
empty). ]]\
[[?look:overlay "overlay": each section under its
current registration, with the atlas borders and any other
atlas_layers on it. ]][[?look:positioning "positioning": the sections and the \
atlas
along the slicing axis as ABBA lays them out: the atlas at
positions_mm above a millimetre ruler, the sections below it in
order of position, each placed section joined by a line to its
position. Unplaced sections appear at the right, labelled "no
position", without a line to the ruler. A long stack is split
into several pictures, none shrunk.]]""",
        ),
        "sections": Argument(
            list[str],
            [],
            """\
Filenames (the extension may be left off); empty is every
section.""",
        ),
        "positions_mm": Argument(
            list[float],
            [],
            """\
Atlas positions in millimetres along the slicing axis:
[[?look:atlas the planes "atlas" draws (at least one)\
[[?look:positioning , and ]]]]\
[[?look:positioning the atlas pictures
above the ruler in "positioning" (none when empty)]].""",
            needs="look:atlas|look:positioning",
        ),
        "images": Argument(
            list[str],
            [],
            """\
Section images to draw, by name (status lists them): one image
alone, in gray, or up to 6 raw channels together, each in its own
colour with automatic contrast. Empty draws `preprocessed`.""",
        ),
        "atlas_layers": Argument(
            list[str],
            [],
            """\
Any of the atlas's images (the job statement lists
them: "template" and any other reference images the atlas ships,
by their names) and "borders". Empty is the mode's default: \
[[?look:overlay borders in "overlay"]]\
[[?look:overlay,look:atlas|look:positioning ,
]]\
[[?look:atlas,look:positioning template in "atlas" and "positioning"||\
[[?look:atlas template in "atlas"]]\
[[?look:positioning template in "positioning"]]]].\
[[?look:section  "section" draws no atlas.]]""",
            needs="look:overlay|look:atlas|look:positioning",
        ),
        "atlas_opacity": Argument(
            float,
            0.5,
            """\
0 to 1: how strongly an atlas image shows under the
section in "overlay".""",
            needs="look:overlay",
        ),
        "warp": Argument(
            Warp,
            "applied",
            """\
"applied" draws each section with its deformation, "none" with
its linear placement only.""",
            needs="look:overlay",
        ),
        "resolution": Argument(
            int,
            0,
            """\
The long edge of each picture in pixels; 0 is the
default size. A section picture is never drawn larger than its
source, an atlas plane at most four times larger.""",
            offered=Offered.SIZE_CHOSEN,
        ),
        "layers": Argument(
            list[str],
            [],
            """\
Names of the layers left on the sections (status lists them),
or ["all"], drawn as marks and outlines on the [[#look:section="section"||\
look:overlay="overlay"]] pictures. The reply says which layers
were drawn. Empty draws none.""",
            needs="look:section|look:overlay",
        ),
        "depth": Argument(
            int,
            OVERLAY_DEPTH,
            """\
Atlas ontology depth whose region boundaries "borders"
draws (0 = root, as in BrainGlobe; larger is finer).""",
        ),
        "masks": Argument(
            list[str],
            [],
            """\
[[?make_mask Names of masks made with make_mask, outlined where
their weight is 0.5 on the sections drawn, on the [[#look:section="section"||\
look:overlay="overlay"]] pictures. Empty draws none.]]""",
            needs="make_mask",
        ),
        "registrations": Argument(
            dict[str, int],
            {},
            """\
{filename: number}: on the [[#look:section="section"||\
look:overlay="overlay"]] pictures, each section named is drawn
under that earlier registration of its own, by the number its status
row gives[[?restore_registration  (restore_registration lists them)]].
Nothing is written; the caption says which registration it is and
whether the section holds it. With sections empty, only these
sections are drawn. Empty draws the registration each section
holds.""",
            needs="look:section|look:overlay",
        ),
        "sheet": Argument(
            bool,
            False,
            """\
True also tiles the pictures into sheets of up to 16, each
tile one picture under its number, a sheet at most as large as
the largest picture you take. The sheets are what this call
returns; every picture is saved and numbered as without it[[?zoom ,
and zoom takes a box on a sheet and draws that box of the tile's
own picture]].""",
            offered=Offered.UNCAPPED,
        ),
    },
    returns="${pictures}",
    slots={
        "pictures": PerCap(
            capped="""\
Up to 4 pictures, attached in the order `pictures` lists them, each
with its number and caption. Pictures past 4 are listed under
`not_shown` with the call that draws them.""",
            uncapped="""\
Every picture drawn, in the order `pictures` lists them, each with its
number and caption. With sheet, `pictures` lists the sheets (each
with its number, caption and its tiles' pictures and boxes) and
`tiled` the pictures they tile.""",
        ),
    },
)

ZOOM = ToolDescription(
    summary=(
        "Zoom into a pixel box or one atlas region of an earlier picture. "
        "Writes nothing."
    ),
    body=(
        """\
The new picture is as large as the earlier one. In a picture of a
section or of a section with the atlas on it, the section is read again
from its original image file at the file's own resolution, only inside
the box, and drawn with the same section image and atlas
borders; the caption says when the file holds fewer pixels than the
picture. [[?look:positioning A positioning picture's sections are drawn again \
from their
working copies (the file reduced to at most 3072 pixels), and the atlas
from its voxels. ]]The box is drawn as the picture was, even if the stack
has changed since (the reply then says `stale`). The new picture has its
own number, so it can be zoomed again. A dashed green grid on it is
labelled with the pixel numbers of the whole picture the zoom was cut
from, and those numbers are the coordinates of a further zoom's \
box[[?set_landmarks  and
of a set_landmarks endpoint, on the zoom or on the whole picture alike]].""",
        """\
[[?look:overlay After a linear[[?fireants_syn|landmark_triangulation|~
trace_borders  or nonlinear]] fit, use region on its overlay to inspect
a particular structure closely, with only that region's border shown.
This can reveal internal mismatches hidden at whole-section scale and
help you decide whether to adjust the linear alignment[[?fireants_syn|~
landmark_triangulation|trace_borders [[?position_sections , refine that
region nonlinearly, or revisit the section's position|| or refine that ~
region nonlinearly]]||[[?position_sections  or revisit the section's ~
position]]]].]]""",
        """\
Earlier pictures remain available by number. [[?look:overlay Reopen or zoom \
a saved
overlay to compare an intermediate registration with the current result;
its saved placement is preserved. ]]Use undo when the latest change makes
the alignment worse.""",
    ),
    arguments={
        "box": Argument(
            list[float],
            [],
            """\
[x0, y0, x1, y1] in pixels of that picture as it was shown,
from its top-left corner (on a zoom, the numbers its grid prints).\
[[?look:overlay  Give either box or region, not both.]]""",
        ),
        "picture": Argument(
            int,
            0,
            """\
The picture's number; 0 is the newest picture that is not
itself a zoom.${sheet_tiles}""",
        ),
        "region": Argument(
            str,
            "",
            """\
One atlas acronym or numeric id, descendants included;
"CTX:left" or "CTX:right" selects one displayed side. Overlay
pictures only. Crops to the entire region with a small margin
and draws only its border, preserving the saved registration
and section image. An absent region is reported. Leave box
empty when using region.""",
            needs="look:overlay",
        ),
    },
    returns="""\
One picture with its number, caption, `covers` [x0, y0, x1, y1] (the
whole picture's pixels it shows) and `content_box` [x0, y0, x1,
y1], where the drawing lies in the image (a narrow box is padded with
black beside it, the caption below), and `redrawn`: false when
the picture could not be drawn again, so the box was cut from the
saved picture and enlarged, with no more detail than it had.""",
    slots={
        "sheet_tiles": PerCap(
            capped="",
            uncapped="""\
 A box on a sheet (look's sheet) falls on one of its
tiles, and that tile's own picture is zoomed.""",
        ),
    },
)

MAKE_IMAGE = ToolDescription(
    summary=(
        "Make, replace or delete a named section image, from the section's "
        "other images."
    ),
    body=(
        """\
A section image is a grayscale picture of a section that look draws and
the fits read by name: the raw channels of the file, never edited, and the
images made here, each saved under the job folder
(sections/<name>/images/<name>.png; status lists them). A fit reads the
file as it is, so a look at an image is what the fit sees. "preprocessed"
was made when the job opened; make it again, or make others under new
names, as the sections need. The steps run in this order on each source:
n4, denoise, CLAHE; then the weighted blend, brightened to a mid-gray
tissue mean; then contrast, gamma, invert. Undoable.""",
    ),
    arguments={
        "name": Argument(
            str,
            REQUIRED,
            """\
The image's name (not a raw channel's); an existing name is
replaced.""",
        ),
        "sections": Argument(
            list[str],
            [],
            """\
Filenames (the extension may be left off) of the sections to
make it for; empty is every section. A later call can remake it for
some sections alone.""",
        ),
        "weights": Argument(
            dict[str, float],
            {},
            """\
The images to blend and their weights, by name: {"dapi": 1},
{"red": 1}, {"dapi": 1, "gfp": 0.3}, or a made image's name. Empty:
each raw channel weighted by how much of the tissue it lights.""",
        ),
        "clahe_clip": Argument(
            float,
            images.DEFAULT_CLAHE_CLIP,
            """\
Local contrast enhancement (CLAHE) clip limit, 0 (none) to
40.""",
        ),
        "clahe_tiles": Argument(
            int,
            images.DEFAULT_CLAHE_TILES,
            "CLAHE tiles per side, 1 to 32.",
        ),
        "n4": Argument(
            bool,
            False,
            "Correct uneven illumination (N4).",
        ),
        "denoise": Argument(
            bool,
            False,
            "Patch-based denoising.",
        ),
        "contrast": Argument(
            list[float],
            [],
            """\
[low, high], fractions 0 to 1 of the blended image's range,
stretched to the full range; empty leaves it.""",
        ),
        "gamma": Argument(
            float,
            0.0,
            "0.1 to 10; 1 or 0 is linear.",
        ),
        "invert": Argument(
            bool,
            False,
            "Bright as dark.",
        ),
        "delete": Argument(
            bool,
            False,
            """\
True removes the image (from the named sections, or
everywhere); the other arguments are then ignored.""",
        ),
        "view": Argument(
            bool,
            True,
            "True returns the pictures described below; false returns none.",
            needs="look:section",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
The recipe now in force and the section images there are[[?look:section , and \
with
view ${sections_shown}, each drawn as look's "section" mode draws the \
image]].""",
    slots={
        "sections_shown": PerCap(
            capped="""\
up to 4 of the sections (spread over the stack when none are
named)""",
            uncapped=(
                "each of the sections (the whole stack when none are named)"
            ),
        ),
    },
)

GREP_ATLAS = ToolDescription(
    summary=(
        "Look regions up in the atlas hierarchy by acronym, name or id. Text "
        "only."
    ),
    body=(
        """\
When you recognize a structure in the tissue, find its atlas name \
here[[?grep_atlas_view ,
then use grep_atlas_view to check its shape and location at candidate
positions. Confirm the region visually before using it to guide a fit
[[?mark_damage or mark damage]]]].""",
        """\
[[?grep_atlas:points With points, it also says which atlas region is drawn at \
given points of
saved pictures: the region (acronym, name, id) and the regions containing it,
or that the point is outside the atlas. A picture that draws the atlas gives a
region[[?look:overlay : an overlay of a section[[?zoom  or a zoom of one]], \
or an
atlas picture drawn for a section]]; a picture of the section alone does \
not.]]""",
    ),
    arguments={
        "query": Argument(
            str,
            "",
            """\
Text matched case-insensitively against region acronyms and
names (substring), or an exact acronym or numeric id[[?grep_atlas:points ;
empty when only points are given]].""",
        ),
        "section": Argument(
            str,
            "",
            """\
Optional filename (the extension may be left off) of a
section with a position; each row then says whether the region
(or any descendant) appears in the atlas plane at that section's
placement.""",
        ),
        "points": Argument(
            list[PicturePoint],
            [],
            """\
[[?grep_atlas:points Points to look the region up at: {picture, x, y}, at
most 40, x and y in pixels from the top-left of the whole picture\
[[?zoom  (on a zoom, the numbers its grid prints)]].]]""",
            needs="grep_atlas:points",
        ),
    },
    returns="""\
Rows of acronym, id, name, ancestry (root to parent, as acronyms)
and descendant count, at most 40, with the number left \
over[[?grep_atlas:points ; and,
per point, its region and containing regions]].""",
)

GREP_ATLAS_VIEW = ToolDescription(
    summary=(
        "Draw the atlas at given positions with chosen regions' borders "
        "highlighted. Writes nothing."
    ),
    body=(
        """\
Only the named regions' borders are drawn, at the stack's cutting
angles. Look highlights no regions; this tool does.""",
        """\
[[?grep_atlas After finding a region with grep_atlas, compare its shape and \
location
here with the structure you recognize in the tissue, before using it
to guide a fit[[?mark_damage  or mark damage]].]]""",
    ),
    arguments={
        "regions": Argument(
            list[str],
            REQUIRED,
            """\
Acronyms, names or ids, descendants included; "CTX:left" or
"CTX:right" names one side, as the pictures show it.""",
        ),
        "positions_mm": Argument(
            list[float],
            REQUIRED,
            """\
Atlas positions in millimetres along the slicing axis;
a position outside the atlas is clamped into its range.""",
        ),
    },
    returns="""\
${pictures}, and per position
the named regions that have no pixel in that plane.""",
    slots={
        "pictures": PerCap(
            capped="Up to 4 pictures with their numbers and captions",
            uncapped="One picture per position with its number and caption",
        ),
    },
)

STATUS = ToolDescription(
    summary=(
        "The stack as it stands, and what this run lets you change. Writes "
        "nothing."
    ),
    returns="""\
One row per section in stack order: filename, position, distance
to the next placed section, flip and quarter turn, transform (kind,
rotation_deg, scale_x, scale_y, shear, translate_x_mm, translate_y_mm,
mirrored), damaged regions and damage note, deformation (the method
that made it, every step's method, and the settings it was made
with), the number of the registration it holds[[?make_mask , masks
(name, source, and whether it was made at an earlier registration)]]. Also the
cutting angles, the interval breaks, the raw channels and the section
images made (each with its recipe, or a script's note, and the
sections it differs on or exists on), the background work still
running, and what this run lets you change.""",
)

LIST_FILES = ToolDescription(
    summary="List the files and folders of the job folder. Writes nothing.",
    body=(
        """\
The job folder keeps every picture you were shown, indexed in
views.jsonl (number, tool, history step, sections, caption), with the
run notes (state.json), the settings (job.json) and the results.""",
    ),
    arguments={
        "path": Argument(
            str,
            ".",
            """\
A folder inside the job folder, relative to it; "." is the job
folder itself.""",
        ),
        "glob": Argument(
            str,
            "",
            """\
Every file below path whose name or path matches this glob
("*.json", "views/*"; "*" is every file below path). Empty lists
path itself.""",
        ),
    },
    returns="""\
The entries, folders first, each with its kind and size, at most 200,
with the number left out.""",
)

SEARCH_FILES = ToolDescription(
    summary=(
        "Search the job folder's text files for lines that match. Writes "
        "nothing."
    ),
    arguments={
        "query": Argument(
            str,
            REQUIRED,
            """\
A regular expression, or plain text when it is not one; upper
and lower case alike.""",
        ),
        "path": Argument(
            str,
            ".",
            "A folder or file inside the job folder, relative to it.",
        ),
        "glob": Argument(
            str,
            "",
            "Only files whose name or path matches this glob.",
        ),
    },
    returns="""\
Matches as "file:line: text", at most 60, with the number left out.
Pictures and binary files are not searched.""",
)

READ_FILE = ToolDescription(
    summary=(
        "Read a text file of the job folder, with line numbers. Writes "
        "nothing."
    ),
    body=(
        """\
A picture file is answered with its entry in the picture index (its
number, tool, sections and caption), never its pixels[[?zoom : zoom shows it
again by that number]].""",
    ),
    arguments={
        "path": Argument(
            str,
            REQUIRED,
            "The file, relative to the job folder.",
        ),
        "offset": Argument(
            int,
            0,
            "Lines to skip from the start.",
        ),
        "limit": Argument(
            int,
            400,
            "Lines to show, at most 2000.",
        ),
    },
    returns="""\
The lines shown, the file's line count and, when lines remain, the
offset to continue from.""",
)

DESCRIPTIONS: dict[str, ToolDescription] = {
    "look": LOOK,
    "zoom": ZOOM,
    "make_image": MAKE_IMAGE,
    "grep_atlas": GREP_ATLAS,
    "grep_atlas_view": GREP_ATLAS_VIEW,
    "status": STATUS,
    "list_files": LIST_FILES,
    "search_files": SEARCH_FILES,
    "read_file": READ_FILE,
}
