# Tool declarations: full (all tasks, image model on)

## look

Draw pictures of sections and the atlas. Writes nothing.

The same request draws the same kind of picture at every stage of the
run: no default depends on the stack's state. Each picture gets a number
and a caption stating the positions, cutting angles, scale and section
image it was drawn with; zoom takes the number.

Use positioning mode to compare sections with candidate atlas planes
along the slicing axis. Use overlays to assess in-plane alignment and
deformation against internal anatomy.

Args:
    mode: "section": each section alone, as oriented, the whole
        image. "atlas": the atlas plane alone at each of positions_mm,
        at the stack's cutting angles; or, for each section named, at that
        section's own position and cutting angles (positions_mm then
        empty). "overlay": each section under its
        current registration, with the atlas borders and any other
        atlas_layers on it. "positioning": the sections and the atlas
        along the slicing axis as ABBA lays them out: the atlas at
        positions_mm above a millimetre ruler, the sections below it in
        order of position, each placed section joined by a line to its
        position. Unplaced sections appear at the right, labelled "no
        position", without a line to the ruler. A long stack is split
        into several pictures, none shrunk.
    sections: Filenames (the extension may be left off); empty is every
        section.
    positions_mm: Atlas positions in millimetres along the slicing axis:
        the planes "atlas" draws (at least one), and the atlas pictures
        above the ruler in "positioning" (none when empty).
    images: Section images to draw, by name (status lists them): one image
        alone, in gray, or up to 6 raw channels together, each in its own
        colour with automatic contrast. Empty draws `preprocessed`.
    atlas_layers: Any of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names) and "borders". Empty is the mode's default: borders in "overlay",
        template in "atlas" and "positioning". "section" draws no atlas.
    atlas_opacity: 0 to 1: how strongly an atlas image shows under the
        section in "overlay".
    warp: "applied" draws each section with its deformation, "none" with
        its linear placement only.
    layers: Names of the layers left on the sections (status lists them),
        or ["all"], drawn as marks and outlines on the "section" and "overlay" pictures. The reply says which layers
        were drawn. Empty draws none.
    depth: Atlas ontology depth whose region boundaries "borders"
        draws (0 = root, as in BrainGlobe; larger is finer).
    masks: Names of masks made with make_mask, outlined where
        their weight is 0.5 on the sections drawn, on the "section" and "overlay" pictures. Empty draws none.
    registrations: {filename: number}: on the "section" and "overlay" pictures, each section named is drawn
        under that earlier registration of its own, by the number its status
        row gives (restore_registration lists them).
        Nothing is written; the caption says which registration it is and
        whether the section holds it. With sections empty, only these
        sections are drawn. Empty draws the registration each section
        holds.

Returns:
    Up to 4 pictures, attached in the order `pictures` lists them, each
    with its number and caption. Pictures past 4 are listed under
    `not_shown` with the call that draws them.


## zoom

Zoom into a pixel box or one atlas region of an earlier picture. Writes nothing.

The new picture is as large as the earlier one. In a picture of a
section or of a section with the atlas on it, the section is read again
from its original image file at the file's own resolution, only inside
the box, and drawn with the same section image and atlas
borders; the caption says when the file holds fewer pixels than the
picture. A positioning picture's sections are drawn again from their
working copies (the file reduced to at most 3072 pixels), and the atlas
from its voxels. The box is drawn as the picture was, even if the stack
has changed since (the reply then says `stale`). The new picture has its
own number, so it can be zoomed again. A dashed green grid on it is
labelled with the pixel numbers of the whole picture the zoom was cut
from, and those numbers are the coordinates of a further zoom's box and
of a set_landmarks endpoint, on the zoom or on the whole picture alike.

After a linear or nonlinear fit, use region on its overlay to inspect
a particular structure closely, with only that region's border shown.
This can reveal internal mismatches hidden at whole-section scale and
help you decide whether to adjust the linear alignment, refine that
region nonlinearly, or revisit the section's position.

Earlier pictures remain available by number. Reopen or zoom a saved
overlay to compare an intermediate registration with the current result;
its saved placement is preserved. Use undo when the latest change makes
the alignment worse.

Args:
    box: [x0, y0, x1, y1] in pixels of that picture as it was shown,
        from its top-left corner (on a zoom, the numbers its grid prints). Give either box or region, not both.
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


## make_image

Make, replace or delete a named section image, from the section's other images.

A section image is a grayscale picture of a section that look draws and
the fits read by name: the raw channels of the file, never edited, and the
images made here, each saved under the job folder
(sections/<name>/images/<name>.png; status lists them). A fit reads the
file as it is, so a look at an image is what the fit sees. "preprocessed"
was made when the job opened; make it again, or make others under new
names, as the sections need. The steps run in this order on each source:
n4, CLAHE; then the weighted blend, brightened to a mid-gray
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
    contrast: [low, high], fractions 0 to 1 of the blended image's range,
        stretched to the full range; empty leaves it.
    gamma: 0.1 to 10; 1 or 0 is linear.
    invert: Bright as dark.
    delete: True removes the image (from the named sections, or
        everywhere); the other arguments are then ignored.
    view: True returns the pictures described below; false returns none.

Returns:
    The recipe now in force and the section images there are, and with
    view up to 4 of the sections (spread over the stack when none are
    named), each drawn as look's "section" mode draws the image.


## grep_atlas

Look regions up in the atlas hierarchy by acronym, name or id. Text only.

When you recognize a structure in the tissue, find its atlas name here,
then use grep_atlas_view to check its shape and location at candidate
positions. Confirm the region visually before using it to guide a fit
or mark damage.

With points, it also says which atlas region is drawn at given points of
saved pictures: the region (acronym, name, id) and the regions containing it,
or that the point is outside the atlas. A picture that draws the atlas gives a
region: an overlay of a section or a zoom of one, or an
atlas picture drawn for a section; a picture of the section alone does not.

Args:
    query: Text matched case-insensitively against region acronyms and
        names (substring), or an exact acronym or numeric id;
        empty when only points are given.
    section: Optional filename (the extension may be left off) of a
        section with a position; each row then says whether the region
        (or any descendant) appears in the atlas plane at that section's
        placement.
    points: Points to look the region up at: {picture, x, y}, at
        most 40, x and y in pixels from the top-left of the whole picture (on a zoom, the numbers its grid prints).

Returns:
    Rows of acronym, id, name, ancestry (root to parent, as acronyms)
    and descendant count, at most 40, with the number left over; and,
    per point, its region and containing regions.


## grep_atlas_view

Draw the atlas at given positions with chosen regions' borders highlighted. Writes nothing.

Only the named regions' borders are drawn, at the stack's cutting
angles. Look highlights no regions; this tool does.

After finding a region with grep_atlas, compare its shape and location
here with the structure you recognize in the tissue, before using it
to guide a fit or mark damage.

Args:
    regions: Acronyms, names or ids, descendants included; "CTX:left" or
        "CTX:right" names one side, as the pictures show it.
    positions_mm: Atlas positions in millimetres along the slicing axis;
        a position outside the atlas is clamped into its range.

Returns:
    Up to 4 pictures with their numbers and captions, and per position
    the named regions that have no pixel in that plane.


## status

The stack as it stands, and what this run lets you change. Writes nothing.

Returns:
    One row per section in stack order: filename, position, distance
    to the next placed section, flip and quarter turn, transform (kind,
    rotation_deg, scale_x, scale_y, shear, translate_x_mm, translate_y_mm,
    mirrored), damaged regions and damage note, deformation (the method
    that made it, every step's method, and the settings it was made
    with), the number of the registration it holds, masks
    (name, source, and whether it was made at an earlier registration). Also the
    cutting angles, the interval breaks, the raw channels and the section
    images made (each with its recipe, or a script's note, and the
    sections it differs on or exists on), the background work still
    running, and what this run lets you change.


## list_files

List the files and folders of the job folder. Writes nothing.

The job folder keeps every picture you were shown, indexed in
views.jsonl (number, tool, history step, sections, caption), with the
run notes (state.json), the settings (job.json) and the results.

Args:
    path: A folder inside the job folder, relative to it; "." is the job
        folder itself.
    glob: Every file below path whose name or path matches this glob
        ("*.json", "views/*"; "*" is every file below path). Empty lists
        path itself.

Returns:
    The entries, folders first, each with its kind and size, at most 200,
    with the number left out.


## search_files

Search the job folder's text files for lines that match. Writes nothing.

Args:
    query: A regular expression, or plain text when it is not one; upper
        and lower case alike.
    path: A folder or file inside the job folder, relative to it.
    glob: Only files whose name or path matches this glob.

Returns:
    Matches as "file:line: text", at most 60, with the number left out.
    Pictures and binary files are not searched.


## read_file

Read a text file of the job folder, with line numbers. Writes nothing.

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


## position_sections

Set where sections sit along the slicing axis, and the stack's cutting angles.

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


## interactive_transform

Set sections' orientation and in-plane transform by hand, and show the result.

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


## mark_damage

Mark the atlas regions a section has lost, so that every fit leaves them out.

Mark the regions whose tissue is missing from the section, or lies folded
over other anatomy, such as a whole hemisphere, the olfactory bulb or the
cortex: the fits leave them out, and the anatomy around them fits
cleanly. Tissue that is intact but shifted near its place is not damage;
left unmarked, the warp fits it. Tissue shifted further is not
damage either: move it with landmarks. A marked region is never fitted, so
intact tissue inside a mark goes unfitted too. When most of a large
region is missing, marking the whole region keeps the anatomy around it
safe. Small tears, bubbles, stains, low contrast and folds within an
intact outline are not damage and are not marked. A section is damaged
exactly when it has marked regions. The marked
regions are left out of elastix_affine, fireants_syn and the image model's
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


## make_mask

Make, replace or delete a named mask of sections: which tissue a fit compares.

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
    section_image: The section image to sort ("preprocessed", a raw channel or an image made with make_image); "em" only.
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


## elastix_affine

Fit sections' in-plane affine transforms automatically with elastix.

For a few sections at a time. Elastix refines each section's current
placement (from the identity when it has no transform) by matching a
section image against an atlas image, inner anatomy included. It
adjusts from where the
section is and does not search from scratch: it moves a section by
small amounts and does not turn it over. The section's marked damage
regions are left out of the atlas automatically. A section that holds a
deformation is refused with HAS_DEFORMATION: undo the deformation
first, or fit a warp on it. Each fit is written as the
section's transform, one undoable write for the call.

Use restrict_to when a few clearly recognizable, surviving regions
provide better registration evidence than the whole section. Those
regions guide a whole-section affine transform. Review the wider
overlay afterward as well.

A region too small for elastix is skipped with REGION_TOO_SMALL; use
a larger region or interactive_transform. No fit is written for it.

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
    section_image: The section image to match, by name: "preprocessed", a raw channel, or an image made with make_image.
    atlas_image: One of the atlas's images, by name (the job statement lists
        them and says what each is), or "nissl", the atlas's Nissl image
        where it has one, else its template.
    mask: "none" compares every pixel of the section, or the
        name of a mask of that section made with make_mask: pixels with
        a weight under 0.5 are left out.
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


## fireants_syn

Fit the atlas onto sections with FireANTs SyN, on their current registration.

This is ANTs SyN (symmetric normalization; Avants et al. 2008, Medical Image
Analysis), the most widely used diffeomorphic registration in neuroimaging,
run by FireANTs (Jena et al. 2026, Nature Communications), which keeps SyN's
model and runs its optimisation on the GPU.

SyN bends the atlas onto a section image of each section, starting
from the section's current registration (its linear placement, and its
deformation when it has one), so each fit builds on the one before;
undo goes back. `restrict_to` refines single regions on top of the
current fit. SyN does not move a piece displaced more than about
400 um from its place; landmarks do. The section's marked damage regions are left out of
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
    section_image: The section image to match, by name: "preprocessed", a raw channel, or an image made with make_image.
    atlas_image: One of the atlas's images, by name (the job statement lists
        them and says what each is), or "nissl", the atlas's Nissl image
        where it has one, else its template.
    mask: "none" compares every pixel of the section, or the
        name of a mask of that section made with make_mask: pixels with
        a weight under 0.5 are left out.
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
    the fold fraction and plausibility flags (tissue folded, regions
    vanished or folded over themselves; displacement outsized
    for the section, measured beyond any landmark warp); the
    `inputs` it used (section image, atlas image as resolved, mask, damage
    regions left out, intensity scaling) and `estimated` (the fraction of
    the section the fit left out on its own); and a
    picture of each section under its new registration with the atlas borders on it. With
    restrict_to, those regions' borders are drawn thick and the others faint, and the
    picture is zoomed to them when they cover a small part of the
    section.


## set_landmarks

Add, update or delete persistent corresponding landmarks. Does not apply a warp.

When a piece of tissue lies away from its place in the atlas outline (a
cerebellum or olfactory bulb pulled off across a gap, a shifted
hemisphere, a folded-back flap), move it with landmarks before a dense
fit: pair points on the piece with their places in the atlas, apply them
with landmark_triangulation, then refine with fireants_syn, which
keeps the landmark move and finishes the fit. Rough pairs are enough.
fireants_syn alone moves such a piece only part of the way and stretches
borders across the gap. A piece within about 400 um of its place needs no
pairs: fireants_syn fits it alone.

Build corresponding pairs across the full section and atlas before calling
landmark_triangulation to apply the set. Pair IDs are stable. Every batch
is one undo step. Atlas and section endpoints are explicitly named.
Coordinates are x,y PIXELS from the top-left of the whole returned picture,
including panel offsets; no fractions or atlas coordinates are accepted.
On a zoom, x,y are the numbers its grid prints (pixels of the whole
picture it was cut from), so a number read off any grid works whether you
name the zoom's picture or the whole picture. Saved geometry resolves the
exact referenced view, including old zooms, orientation and an applied warp.
Caption/gutter/out-of-image points are refused, and so is an atlas endpoint
outside the atlas (the refusal gives the nearest point inside it).
A picture of the
atlas alone supplies atlas points only when drawn with
look(mode="atlas", sections=[section]); generic atlas browsing has no
section context. Plane changes retain pairs but flag their atlas
correspondences stale; update those atlas endpoints.
Prefer both endpoints on the same overlay, or a zoom of it: the
atlas point where an atlas border is drawn, the section point where that
border belongs in the tissue. You point at what you see; a point on a warped
overlay is mapped back to the section itself. Read endpoints off a close
zoom, its grid lines 10 to 20 pixels apart: read off a coarse grid, a point can
miss the border you meant by 20 pixels or more. Spread pairs over the outline and
the internal anatomy: a pair on the outline moves the outline, and a pair
on an internal landmark (the midline, a ventricle, a fibre tract) moves the
anatomy inside. A pair moves the tissue around it, out to the neighbouring
pairs and the image corners, so tissue that is already right takes pairs
too: a pair with both endpoints on the same point (one pixel
of an overlay) keeps that spot where it is.  An atlas region
with no tissue on the section (torn or missing) takes no pair: mark it with
mark_damage; a region
only partly missing keeps its pairs on the part that is present. Tissue with no
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


## landmark_triangulation

Apply this section's saved landmark set as a piecewise affine warp.

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

Returns:
    Landmark count, `built_from` (the pair IDs the warp was made of),
    whether applied/composed,
    displacement_mm (the whole deformation now applied) and
    step_displacement_mm (only the change this application made over the
    warp it replaced), diagnostics and any folds. The picture's
    number is under `pictures`.


## trace_borders

Trace the atlas borders onto a section with the image model, then fit it with FireANTs.

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

Image 1: the photograph of a brain coronal section to edit. Image 2: the same photograph in exactly the same frame, carrying thin yellow region boundaries placed by a rough alignment of the atlas to this specimen; Image 2 alone decides which boundaries exist.

Task: return Image 1 with the boundaries of Image 2 drawn on this specimen's anatomy, using Image 2 as the starting point. Each boundary follows the edge of its region where that edge shows in the photograph; where the region is faint or indistinct, the boundary keeps the position and shape it has in Image 2, fitted to the structures around it. Correct Image 2 two ways. The drawing carries no line around a feature of the slide rather than of the brain, such as a bubble, a stain or debris; the feature itself stays in the photograph as it is. A line that does not hug its region's edge is adjusted by a shift or bend until it does. Cracks and folds are likewise features of the slide: a boundary that meets one continues along the anatomy beneath it. Tissue that is physically torn away or missing from the section, where slide background shows in place of brain, is the only place a boundary of Image 2 is left out: the part of a boundary that would lie over missing tissue is omitted and the rest of that boundary is drawn; a torn or cut edge is not a region edge and gets no line of its own. A region that is present but faint is not missing, and its boundary is drawn. A piece of tissue that has shifted or turned keeps its boundaries, drawn on the piece where it lies.

This is an annotation overlay on a photograph. Change only by adding the yellow lines; the corrected lines are the only difference from Image 1. Preserve everything else exactly: every tissue pixel and its texture, the background, the brain's size and position, the frame and aspect. The lines are thin bright yellow and form exactly the partition of Image 2 on the surviving tissue: every boundary of Image 2 that lies on tissue, no other line, no fill, no label. Output one image.


## note

Append one line to the run notes, which are saved with the results.

Args:
    text: The note.


## undo

Undo the last write. A tool call, or a finished piece of background work, is one step.

Args:
    steps: How many steps to undo at once, newest first (all the history
        holds when it holds fewer).

Returns:
    The sections undone and the fields that changed, and the rows of
    those sections.


## redo

Redo the write that ``undo`` reversed.

Args:
    steps: How many undone steps to redo at once.

Returns:
    The sections redone and the fields that changed, and the rows of
    those sections.


## restore_registration

Put back registrations sections held before: their orientation, transform and deformation.

Each registration a section holds gets a number, kept whatever is
written after it: the `registration` of its status row and of the reply
to the write that left it. Undo takes back the job's last write; this
returns sections to any earlier registrations, such as fits made before
another method's, and leaves the other sections as they are. One
undoable write for the call. A deformation fitted with the section at
another position or other cutting angles is refused until the section is
back there. look draws an earlier registration
without writing it.

Args:
    section: Filename (the extension may be left off) of a section whose
        registrations to list; writes nothing. Give section or sections.
    sections: The registrations to put back: {id, registration} per
        section, id a filename and registration a number from the list.
    view: Return a picture of each section put back, under the
        registration put back.

Returns:
    With section, its registrations: per number the tool that made it and
    when, its transform (kind, rotation_deg, scale_x, scale_y, shear,
    translate_x_mm, translate_y_mm), its deformation (the method that made
    it, every step's method, and the settings it was made with), the
    numbers of the pictures saved with the write that left it, and which
    one the section holds. With sections, per section the registration
    it holds now and the one it replaced, or why it was not put back;
    the rows of the sections written, and the pictures.


## submit

End the run. Call this exactly once, last.

Waits for the background work still running. Refused, with the reason
and nothing changed, while something the run needs is missing.

Args:
    summary: One or two sentences on what you did.
    notes: Short observations worth carrying forward.
    interval_breaks: Filenames (the extension may be left off) of the
        sections AFTER a gap you conclude is real. Empty if there are
        none.


## stalign_lddmm

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
has one) and refines it; it does not search from scratch: set the orientation and a rough transform with interactive_transform first. With fit "warp" the transform stays and
only the bend is fitted, on top of the deformation the section holds, a landmark warp included, so the landmark move is kept. With fit "both"
it also refits the section's in-plane transform and writes it; refused on a
section that holds a deformation, since a new transform clears it. The
section's marked damage regions are left out automatically; mark them with mark_damage first.

STalign takes no section mask: it sorts the section's pixels itself. Before
each comparison every pixel is weighted between matching (compared),
background (slide, ignored) and artifact (debris, ignored), from the
mismatch and from the background and artifact means (sorting, below). The
final sorting is saved with the deformation, and make_mask can take it as a mask for another fit (source fit). A named mask is refused
(MASK_UNSUPPORTED); the engines that take one are fireants_syn, elastix_affine, emlddmm_stack and spomialign. STalign fits the whole
section: it takes no regions; refine single regions afterwards with fireants_syn, which builds on this fit. It keeps its own brightness
scaling: slide 0, typical tissue 1.

Starts one piece of background work per section and returns at once with
their ids, so you can carry on with other sections. A write to a section
whose fit still runs is refused until it lands. When a section's fit
lands, the next tool reply starts with its notice: what was fitted, the
displacement, fold fraction and flags, what STalign used and estimated,
and the file in the job folder that holds STalign's own mismatch record
for the fit, which read_file opens; then its pictures. Each
landing is its own undo step. status lists the work still running; submit
waits for it. Judge the overlay against the internal anatomy.

Args:
    sections: One to twenty filenames (the extension may be left off).
    section_image: The section image to fit, by name: "preprocessed", a raw channel, or an image made with make_image. STalign reads it as grey.
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
    sorting: {"method": "classic" | "em" | "none"} and any of "match_sd",
        "background_sd", "artifact_sd" (spreads), "background_mean",
        "artifact_mean" (left out: estimated) and "artifact" (true adds the
        artifact bin to "em"). Brightness units: slide 0, typical tissue 1.
        "classic" is STalign's published sorting with fixed spreads; "em"
        measures the matching spread itself and takes the background mean
        from the slide at the image border, for sections with missing
        tissue; "none" compares every pixel. Default {"method": "classic"}
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
    and when it set nothing aside the notice says so and sends no picture.


## stalign_plane_search

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
at the found plane looks wrong, and write the pairwith position_sections. Keep the positions you placed unless a
section's picture shows its position change is right. A section mounted
mirrored reverses the sign of its yaw and keeps its pitch, and the
hemispheres often cannot be told apart by any other sign, so many
sections of a stack, up to half, may be mirrored: the reply sorts the
sections by the sign of their yaw. Make every section agree on one sign
(mirror the smaller group with interactive_transform and search them again) before writing the angles.

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
    written.


## spomialign

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
the section's current placement and does not search from scratch: place the section with position_sections and set its flip, quarter turn and rough transform with interactive_transform first. The section's marked damage regions are left
out automatically; where tissue is missing or lies folded over other anatomy, mark those regions with mark_damage first (tissue that is intact but shifted is not damage).

fit chooses what is fitted:
- "transform": the section's transform: rotation, shift, scale and shear
  (an affine), or with rigid rotation and shift alone, the section keeping
  its size. Written as the section's transform.
- "warp": a smooth local deformation on top of the registration the section
  holds (its transform, and its deformation when it has one, a landmark warp included, so the landmark move is kept); the transform
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
    section_image: The section image to match, by name: "preprocessed", a raw channel, or an image made with make_image. Default "preprocessed".
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names), or "nissl", the atlas's Nissl (its template when
        it ships none). Default "nissl"; the reply names the image used.
    mask: "none", or the name of a mask of each section (made with make_mask); matches ending where it is below one half
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
    matched pairs as a table and a picture, which read_file opens); and a picture of each section under its new registration
    with the atlas borders on it. SPOmiAlign estimates no weighting, so
    estimated is null.


## emlddmm_stack

Fit the whole stack to the atlas at once with emlddmm, on its current plane.

emlddmm (Tward et al. 2025, Nature Communications) maps whole-brain datasets
of any modality into the Allen CCF. It models every section as an image of one
deformed 3-D atlas, with an EM sorting of missing tissue and artifacts and a
contrast fit in small blocks.

Use this when you have most of a brain's series, closely and evenly
spaced, and want placements that agree across the stack and fit the
brain's overall shape. It bends the brain as a whole, never one section
alone; for a local mismatch on one section (a ventricle, a displaced piece) use a per-section warp.

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
refined from its current registration, its deformation included, so a landmark move is kept. The section's marked damage regions are
left out automatically; mark them with mark_damage first.

emlddmm takes a section mask as soft weights: the mask multiplies each
pixel's weight in the fit, so a pixel at 0.3 counts at 0.3. Besides the
mask, emlddmm estimates its own sorting during the fit; the final sorting
of each section it writes a deformation for is saved with that
deformation, and make_mask can take it as a mask for another fit (source fit). It keeps its own brightness scaling: the section image as it
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
you can look, zoom and work on sections you left out. When it lands, the
next tool reply starts with its notice: how many sections were written,
those fitted for the stack and not written with the reason, the sections
whose warp has plausibility flags with their codes, and the overlays of
up to four sections (flagged ones first, then those that moved most) with
how far each moved. Judge the rest with look. emlddmm's own energy record,
the images it saw and the per-section results (movement, warp
displacement, fold fraction, flags, tissue set aside) are files in the
job, which read_file opens.

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
    section_image: The section image to fit, by name: "preprocessed", a raw channel, or an image made with make_image. Read as grey, as it
        is, 0 to 1. Default "preprocessed".
    atlas_image: One of the atlas's images (the job statement lists
        them), or "nissl", the atlas's Nissl (its template when it ships
        none). Default "nissl"; the reply names the image used.
    mask: "none", or the name of a mask that every section in the stack
        has (made with make_mask); its weights multiply the
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
    estimated (its sorting and the share of the tissue it left out).


## emlddmm_plane

Estimate the stack's cutting angles and spacing with emlddmm. Writes nothing.

Fits emlddmm's 3-D affine to the whole placed stack at once, with the
plane free and no warp: every section votes on one pitch and yaw, and
the spacing may stretch and shift as a whole (a section cannot move on
its own). Starts from the current angles and positions, so place and
orient every section first, with the atlas outline over its tissue (emlddmm
cannot recover from a start 1 to 3 mm off); one call is one descent and lands short
from far away, so write the angles with position_sections and run it again until two passes agree within a
degree or two. emlddmm cannot flip a section: a section mounted mirrored
and not yet flipped pulls the yaw toward zero and stands out from the
others in its residual turn or shift here; flip it with interactive_transform and run again before writing. Keep the
positions you placed unless the spacing change looks right in the
pictures. emlddmm's energy record and the images it saw are files in the
job, which read_file opens.

Args:
    sections: Filenames; empty means every section with a position. At
        least two.
    slot_um: The slot spacing in micrometres, as for emlddmm_stack: you
        set it, there is no default. With the order right and each section
        within about one interval of its place, set it to the section
        thickness, or to about half the interval between neighbouring
        sections.
    section_image: The section image to fit, as for emlddmm_stack.
    atlas_image: One of the atlas's images or "nissl", as for the fit.
    iterations: Steps at the coarsest scale, as for emlddmm_stack. Default
        100.
    view: True returns the pictures described below; false returns none.

Returns:
    pitch_deg, yaw_deg and their change from the stack's angles; the
    spacing scale and shift_mm; per section what the fit used (inputs), its
    proposed position_mm, the
    change, and the residual turn (deg) and shift (mm) it needed at the
    found plane beyond what the stack as a whole needed; and the overlays of up to four sections drawn at the found plane, not written.
