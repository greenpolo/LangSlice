# Tool declarations: host (forced pictures, one tool off, deformation required)

## look

Draw pictures of sections and the atlas. Writes nothing.

The same request draws the same kind of picture at every stage of the
run: no default depends on the stack's state. Each picture gets a number
and a caption stating the positions, cutting angles, scale and channel
settings it was drawn with; zoom takes the number.

Use positioning mode to compare sections with candidate atlas planes
along the slicing axis. Use overlays to assess in-plane alignment and
deformation against internal anatomy.

Args:
    mode: "section": each section alone, as oriented, the whole
        image. "atlas": the atlas plane alone at each of positions_mm,
        at the stack's cutting angles, or with one section at that
        section's own plane (positions_mm may be empty). "overlay": each section under its
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
    channels: The raw channels of each section to draw, by name (several
        are blended in distinct colours, at most 6), or one derived
        channel alone: ["preprocessed"] for the channel the fits and the
        image model read, or the name of another channel
        made with set_channel. Empty is every raw channel, each with its
        display settings.
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
    masks: Names of masks made with set_mask, outlined where
        their weight is 0.5 on the sections drawn, on the "section" and "overlay" pictures. Empty draws none.

Returns:
    Up to 4 pictures, attached in the order `pictures` lists them, each
    with its number and caption. Pictures past 4 are listed under
    `not_shown` with the call that draws them.


## zoom

Zoom into a pixel box or one atlas region of an earlier picture. Writes nothing.

The new picture is as large as the earlier one. In a picture of a
section or of a section with the atlas on it, the section is read again
from its original image file at the file's own resolution, only inside
the box, and drawn with the same channels, display settings and atlas
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
        and display settings. An absent region is reported. Leave box
        empty when using region.

Returns:
    One picture with its number, caption, `covers` [x0, y0, x1, y1] (the
    whole picture's pixels it shows) and `content_box` [x0, y0, x1,
    y1], where the drawing lies in the image (a narrow box is padded with
    black beside it, the caption below), and `redrawn`: false when
    the picture could not be drawn again, so the box was cut from the
    saved picture and enlarged, with no more detail than it had.


## set_channel

Set how a channel is displayed, or the recipe of a channel made from the raw ones.

A raw channel is a channel of the image files, named as the reply's
`channels` lists them; it is never edited, and setting it changes only how
look draws it (contrast_limits, gamma, colormap): nothing a fit or the
image model reads. Every other name is a derived channel, made from the
raw ones by a recipe (weights, clahe_clip, clahe_tiles, n4, denoise,
contrast_limits, gamma, invert) for the whole stack, or for the
sections named as their own recipe. "preprocessed" is the derived channel
the fits and the image model read by default: LangSlice's default recipe
until this is called; a new name makes another channel that the
fits and look can read as well. The setting stays until it is
changed, and every caption states it. Undoable; another call replaces the
recipe.

Register on the counterstain. A counterstain (DAPI, Hoechst, Nissl,
NeuroTrace) labels every cell, so it shows the anatomy the atlas draws; a
labelling channel (a tracer, an antibody, a reporter) marks only some
cells, and its bright signal pulls the fits toward it. Give the
counterstain weight 1 and each labelling channel weight 0, then enhance
it until the internal anatomy (layers, nuclei, fibre tracts, ventricles)
is as clear as possible: CLAHE always, N4 where illumination is uneven,
denoising where it is grainy. A section with one channel, or a
brightfield stain that already shows the anatomy, uses that channel. The
preprocessed channel serves both you and the fitting algorithms: look at
it (`look` with channels ["preprocessed"]) to judge alignment, and the
fits match it against the atlas. Pale or low-contrast sections
especially need this before you place or fit them. Inspect the
before-and-after pictures and compare with the raw channels before
fitting. A raw channel's display adjustments alone do not change what the
fitting algorithms receive.

Args:
    name: A raw channel's name, "preprocessed", or the name of a derived
        channel to make or change.
    sections: Filenames (the extension may be left off) of the sections
        that get a recipe of their own; empty sets the whole stack's
        recipe. A raw channel is set for every section that has it.
    weights: A derived channel's raw channels and their weights, by name
        (for example {"dapi": 1, "gfp": 0}); a channel left out or at 0
        is not used. Empty: automatic weights by tissue coverage.
    clahe_clip: CLAHE clip limit, 0 (no CLAHE) to 40.
    clahe_tiles: CLAHE tiles per side, 1 to 32.
    n4: N4 correction of uneven illumination.
    denoise: Denoising.
    contrast_limits: [low, high]. For a raw channel, in the file's own
        intensities: low is drawn black, high at full brightness; empty
        keeps the current limits, and without limits a channel is drawn
        with automatic ones, per section: black at the background, full
        brightness at the tissue's 95th percentile. For a derived channel,
        [low, high] between 0 and 1 of the recipe's result, stretched to
        the full range; empty leaves it as it is.
    gamma: 0.1 to 10; 1 is linear. 0 keeps the current gamma (a derived
        channel's: none).
    colormap: A raw channel's: "gray", "red", "green", "blue", "magenta",
        "cyan" or "yellow"; empty keeps the current one.
    invert: A derived channel's: True draws bright as dark.
    reset: True returns a raw channel to its default display, "preprocessed"
        to the default recipe (the stack's, or the named sections' back to
        the stack's), and deletes another derived channel (with sections,
        only those sections' recipes of its own); the other arguments are
        then ignored.

Returns:
    The settings or recipe now in force, the channels there are, and for
    a derived channel, up to 4 of the sections (spread over the stack when
    none are named) two pictures each: the channel before this call, then
    after it; a channel this call creates has no before, so one picture each.


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
    to the next placed section, flip and quarter turn, transform,
    damaged regions and damage note, deformation steps, masks
    (name, source, and whether it was made at an earlier registration). Also the
    cutting angles, the interval breaks, the raw channels' display
    settings ("auto": automatic contrast limits), the derived channels'
    recipes (the preprocessed channel's, and any other you made), the
    background work still running, and what this run lets you change.


## list_files

List the files and folders of the job folder. Writes nothing.

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


## set_mask

Make, replace or delete a named mask of one section: which tissue a fit compares.

A mask holds a weight from 0 to 1 for each pixel of the section; a fit
that takes it by name ignores the pixels under 0.5. It sorts the
section's pixels into tissue, artifact and background by their brightness
in a channel and by where the atlas puts tissue, as the fits' own estimate
does, and it is the only way a fit leaves out a hole, a tear or a stain.
The mask sits on the section's own pixels, so it stays usable after
fits and moves; status says when it was made at an earlier registration.
Turning or flipping the section makes a fit refuse it until it is made
again. The marked damage regions are not part of a mask; every
fit leaves them out of the atlas on its own. Undoable.

Args:
    section: A filename (the extension may be left off).
    name: The mask's name; an existing name is replaced.
    source: "em": sort the section's pixels now, at its current
        registration, against atlas_image. "fit": the sorting the
        section's applied deformation saved with it, copied as it was
        (NO_DEFORMATION or NO_SORTING when it has none).
    channel: The section channel to sort (a raw channel, "preprocessed" or one you made with set_channel); "em" only.
    atlas_image: The atlas image the sorting compares with ("nissl" is the
        atlas's Nissl image where it has one, else its template); "em"
        only.
    sorting: Starting points of the sort, each between 0 and 1 of the
        channel's brightness or a spread in the same units: match_sd,
        background_sd, artifact_sd, background_mean, artifact_mean. Empty
        uses the defaults; "em" only.
    delete: True removes the mask; the other arguments except section and
        name are then ignored.

Returns:
    The mask's source, channel, atlas image and starting points, and the
    fraction of the section's tissue it leaves out. With view, the
    section drawn as look's "section" mode draws it, with the mask's
    0.5 outline on it, and a caption saying the outlines trace the mask's
    edge, that the ringed patches are what the mask leaves out of a fit,
    and the mask's name and source.


## elastix_affine

Fit sections' in-plane affine transforms automatically with elastix.

For a few sections at a time. Elastix refines each section's current
placement (from the identity when it has no transform) by matching a
channel of the section against an atlas image, inner anatomy included. It
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
atlas outline to 0-1. The reply names the channel, atlas image and mask each
fit used.

Args:
    sections: One to four filenames (the extension may be left off).
    restrict_to: Regions to fit by (acronyms, names or ids, descendants
        included; "CTX:left" or "CTX:right" for one side): only the
        atlas within 300 um of them, against the tissue the fit lays
        there. Only the ANTs and Elastix fits take it. Empty fits by
        every region.
    section_image: The section channel to match: "preprocessed" (the channel
        the fits and the image model read by default), a raw channel's name or a channel you made with set_channel.
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names) or "nissl", the atlas's Nissl image where it has
        one, else its template.
    mask: "none" compares every pixel of the section, or the
        name of a mask of that section made with set_mask: pixels with
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

SyN bends the atlas onto a channel of each section, starting
from the section's current registration (its linear placement, and its
deformation when it has one), so each fit builds on the one before;
undo goes back. Fit the whole section first, then refine regions with
restrict_to. A piece displaced more than about 400 um from its
place is moved with landmarks first. The section's marked damage regions are left out of
the atlas automatically. Needs a position and a transform. Applied as the
section's deformation, one undoable write; a later change to the
section's position, orientation, cutting angles or transform clears it.

Use restrict_to when a few clearly recognizable, surviving regions
provide better registration evidence than the whole section, to refine
the existing registration locally. Review the wider overlay afterward
as well.

Both images are scaled from their 1st to 99th percentile inside the
atlas outline to 0-1. The step size and the number of iterations at each
scale are fixed. The reply names the channel, atlas image and mask each fit
used.

Args:
    sections: One to four filenames (the extension may be left off).
    restrict_to: Regions to fit by (acronyms, names or ids, descendants
        included; "CTX:left" or "CTX:right" for one side): the atlas and
        the section are both cut to the regions' neighbourhood and the
        displacement fades to nothing beyond it. Only the ANTs and
        Elastix fits take it. Empty fits by every region.
    section_image: The section channel to match: "preprocessed" (the channel
        the fits and the image model read by default), a raw channel's name or a channel you made with set_channel.
    atlas_image: One of the atlas's images (the job statement lists
        them: "template" and any other reference images the atlas ships,
        by their names) or "nissl", the atlas's Nissl image where it has
        one, else its template.
    mask: "none" compares every pixel of the section, or the
        name of a mask of that section made with set_mask: pixels with
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

Add, update or delete persistent corresponding landmarks and holds. Does not apply a warp.

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
Caption/gutter/out-of-image points are refused. A picture of the
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
anatomy inside. Place no pair on a spot just to keep it where it is:
landmark_triangulation adds automatic holds between the pairs, so a pair
moves its neighbourhood up to the nearby holds and tissue far from every pair
stays where it was. A hold of your own is one section point that
landmark_triangulation keeps where it is, with or without the automatic
holds: place holds on anatomy that is already right, such as the main tissue
beside a piece whose pairs must move it across a gap. Tissue with no atlas counterpart (debris, or tissue from
another section) takes no pair either. With no pairs, holds or delete, nothing
is written and the reply lists the pairs and holds.

Args:
    section: Section filename (the extension may be left off).
    pairs: New pairs: {atlas: {picture, x, y}, section: {picture, x, y}}.
        Updates add id and either or both endpoints; omitted endpoints stay.
        At most four added/edited pairs with view true.
    holds: New holds: {picture, x, y}, a point on a picture of the
        section, as a pair's section endpoint. IDs H1, H2, ...; a hold is
        not edited: delete it and add another.
    delete: Stable pair and hold IDs to delete from this section, or
        ["all"] to delete every pair and hold of this section.

Returns:
    The changed and deleted IDs; for each pair given an atlas endpoint, the
    region drawn there (`atlas_endpoint_regions`), with a warning naming
    any atlas endpoint outside the atlas; and every pair of the section, one record
    each: id, atlas and section as {picture, x, y}, and state: applied (in
    the landmark warp applied now, as they are), moved (in it, but at other
    endpoints now), pending (saved, not in it) or stale (placed against an
    earlier atlas plane); with the count of each; every hold (`holds`): id,
    section as {picture, x, y}, and state applied or pending, the new ones
    under `added_holds`; and `warp`: the pairs and holds the current warp
    was built from, those not in it, those deleted but still shaping it,
    and that landmark_triangulation must run again to apply the set as it
    stands.


## landmark_triangulation

Apply this section's saved landmark set as a piecewise affine warp.

Requires at least three corresponding pairs, with coverage across the full
section; the tool cannot assess anatomical coverage. Before it builds the
mesh it adds automatic holds between the pairs: a regular lattice of spots over
the tissue, left out near any pair, each kept where it currently is (its
current placement: linear, or the earlier deformation this composes onto). A
pair therefore moves its neighbourhood up to the nearby holds, and tissue far
from every pair stays where it was. Automatic holds are not pairs: they are
listed nowhere, cannot be edited, and are recomputed from the pairs on every
application; the picture draws them as small unlabeled dots. Turn them off
(holds "none") when a piece lies across a gap of empty slide from its
place: the automatic holds land on the tissue beside the piece, and its
pairs fold the mesh pulling past them; keep the tissue that is already right
with holds of your own (set_landmarks holds), placed off the piece's path to
its place, which are kept where they are with either choice and drawn as
labelled squares. Uses VisuAlign's
section-domain Delaunay mesh in pixels over the pairs and holds, and
four fixed corners 10% outside the section. Repeated section points,
zero-area triangles and stale atlas planes are refused. Folded triangles are
applied, as VisuAlign does: the reply gives the share of the section they
cover and the pairs at their corners (a hold of yours at a corner is named by
its ID, an automatic hold as such), and the picture outlines them in magenta.
The inverse uses the same triangles (unfolded ones first where folded ones
overlap). Composes a correction onto the current deformation; undo removes
the application. Editing pairs or holds alone never warps: pairs added,
moved or deleted, and holds added or deleted, after an application change
nothing until this runs again.

Args:
    section: Section filename (the extension may be left off).
    holds: "automatic" adds the automatic holds; "none" adds none, so only
        the pairs, your holds and the four fixed corners shape the warp.

Returns:
    Landmark count, `built_from` (the pair IDs the warp was made of), the
    holds used (`holds` as asked, how many automatic holds were added,
    `placed_holds`: your hold IDs) in words (`holds_note`), whether
    applied/composed,
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

Returns:
    The sections undone and the fields that changed, and the rows of
    those sections.


## redo

Redo the write that ``undo`` reversed.

Returns:
    The sections redone and the fields that changed, and the rows of
    those sections.


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
