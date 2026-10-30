"""The landmark tools' descriptions: setting pairs and applying them.

Each tool's description is data, read as
:mod:`~langslice.tools.descriptions.description` says.
"""

from __future__ import annotations

from langslice.tools.arguments import LandmarkPair
from langslice.tools.descriptions.description import (
    REQUIRED,
    Argument,
    Distance,
    Offered,
    ToolDescription,
)

#: How far a displaced piece lies from its place before a dense fit alone no
#: longer reaches it, so landmarks move it first (adult mouse; measured on
#: sagittal and horizontal sections).
LANDMARK_MOVE_UM = 400.0

SET_LANDMARKS = ToolDescription(
    summary=(
        "Add, update or delete persistent corresponding landmarks. Does not "
        "apply a warp."
    ),
    body=(
        """\
When a piece of tissue lies away from its place in the atlas outline (a
cerebellum or olfactory bulb pulled off across a gap, a shifted
hemisphere, a folded-back flap), move it with landmarks before a dense
fit: pair points on the piece with their places in the atlas, apply them
with landmark_triangulation[[?fireants_syn , then refine with fireants_syn, \
which
keeps the landmark move and finishes the fit. Rough pairs are enough.
fireants_syn alone moves such a piece only part of the way and stretches
borders across the gap. A piece within ${landmark_move} of its place needs no
pairs: fireants_syn fits it alone]].""",
        """\
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
[[?look:overlay Prefer both endpoints on the same overlay[[?zoom , or a zoom \
of it]]: the
atlas point where an atlas border is drawn, the section point where that
border belongs in the tissue. You point at what you see; a point on a warped
overlay is mapped back to the section itself. ]][[?zoom Read endpoints off a \
close
zoom, its grid lines 10 to 20 pixels apart: read off a coarse grid, a point \
can
miss the border you meant by 20 pixels or more. ]]Spread pairs over the \
outline and
the internal anatomy: a pair on the outline moves the outline, and a pair
on an internal landmark (the midline, a ventricle, a fibre tract) moves the
anatomy inside. A pair moves the tissue around it, out to the neighbouring
pairs and the image corners, so tissue that is already right takes pairs
too: a pair with both endpoints on the same point[[?look:overlay  (one pixel
of an overlay)]] keeps that spot where it is. [[?mark_damage  An atlas region
with no tissue on the section (torn or missing) takes no pair: mark it with
mark_damage; a region
only partly missing keeps its pairs on the part that is present. ]]Tissue \
with no
atlas counterpart (debris, or tissue from another section) takes no pair \
either.
With no pairs and no delete, nothing is written and the reply lists the \
pairs.""",
    ),
    arguments={
        "section": Argument(
            str,
            REQUIRED,
            "Section filename (the extension may be left off).",
        ),
        "pairs": Argument(
            list[LandmarkPair],
            [],
            """\
New pairs: {atlas: {picture, x, y}, section: {picture, x, y}}.
Updates add id and either or both endpoints; omitted endpoints stay.
At most four added/edited pairs with view true.""",
        ),
        "delete": Argument(
            list[str],
            [],
            """\
Stable pair IDs to delete from this section, or ["all"] to
delete every pair of this section.""",
        ),
        "view": Argument(
            bool,
            True,
            """\
Show one contextual side-by-side crop for each added/edited pair:
atlas TEMPLATE and section PREPROCESSED. False shows no pictures.""",
            offered=Offered.VIEW_CHOSEN,
        ),
    },
    returns="""\
The changed and deleted IDs; for each pair given an atlas endpoint, the
region drawn there (`atlas_endpoint_regions`); and every pair of the section, \
one record
each: id, atlas and section as {picture, x, y}, and state: applied (in
the landmark warp applied now, as they are), moved (in it, but at other
endpoints now), pending (saved, not in it) or stale (placed against an
earlier atlas plane); with the count of each; and `warp`: the pairs the
current warp was built from, those not in it, those deleted but still
shaping it, and that landmark_triangulation must run again to apply the
set as it stands.""",
    slots={"landmark_move": Distance(LANDMARK_MOVE_UM, "about {} um")},
)

LANDMARK_TRIANGULATION = ToolDescription(
    summary=(
        "Apply this section's saved landmark set as a piecewise affine warp."
    ),
    body=(
        """\
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
application change nothing until this runs again.""",
    ),
    arguments={
        "section": Argument(
            str,
            REQUIRED,
            "Section filename (the extension may be left off).",
        ),
        "view": Argument(
            bool,
            True,
            "Show the resulting registration under atlas borders.",
            offered=Offered.VIEW_CHOSEN,
        ),
        "resolution": Argument(
            int,
            0,
            """\
The long edge of the picture in pixels, as for look; 0 is
the default size.""",
            offered=Offered.SIZE_CHOSEN,
        ),
    },
    returns="""\
Landmark count, `built_from` (the pair IDs the warp was made of),
whether applied/composed,
displacement_mm (the whole deformation now applied) and
step_displacement_mm (only the change this application made over the
warp it replaced), diagnostics and any folds. The picture's
number is under `pictures`.""",
)

DESCRIPTIONS: dict[str, ToolDescription] = {
    "set_landmarks": SET_LANDMARKS,
    "landmark_triangulation": LANDMARK_TRIANGULATION,
}
