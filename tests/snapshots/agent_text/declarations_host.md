# Tool declarations: host (forced pictures, one tool off, deformation required)

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

Returns:
    The recipe now in force and the section images there are, and with
    view up to 4 of the sections (spread over the stack when none are
    named), each drawn as look's "section" mode draws the image.


## grep_atlas

Look regions up in the atlas hierarchy by acronym, name or id. Text only.

When you recognize a structure in the tissue, find its atlas name here,
then use grep_atlas_view to check its shape and location at candidate
positions. Confirm the region visually before using it to guide a fit.

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
to guide a fit.

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

Returns:
    Per section its orientation, its transform as the values above,
    and whether anything was written; and a picture of each section
    under its new transform with the atlas borders on it.


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

Returns:
    Per section the displacement (median and max, mm, over the placed atlas),
    the fold fraction and plausibility flags (regions compressed,
    expanded, vanished or folded beyond limits; displacement outsized
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
of an overlay) keeps that spot where it is. Tissue with no
atlas counterpart (debris, or tissue from another section) takes no pair either.
With no pairs and no delete, nothing is written and the reply lists the pairs.

Args:
    section: Section filename (the extension may be left off).
    pairs: New pairs: {atlas: {picture, x, y}, section: {picture, x, y}}.
        Updates add id and either or both endpoints; omitted endpoints stay.
        At most four added/edited pairs with view true.
    delete: Stable pair IDs to delete from this section, or ["all"] to
        delete every pair of this section.

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

Returns:
    Landmark count, `built_from` (the pair IDs the warp was made of),
    whether applied/composed,
    displacement_mm (the whole deformation now applied) and
    step_displacement_mm (only the change this application made over the
    warp it replaced), diagnostics and any folds. The picture's
    number is under `pictures`.


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
