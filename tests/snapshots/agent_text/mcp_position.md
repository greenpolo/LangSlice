# MCP tool list: position job

## start_job

Open the job the user named: an image_folder (or its job folder),
        or a saved ABBA job's job_id.

        Returns the job facts and status table without images. Read every
        show_stack page before writing. Omit both arguments to repeat the
        current briefing. Never supply both job_id and image_folder.
        

- `image_folder` (string):
- `job_id` (string):

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
        

- `mode` (string, required, one of ['section', 'atlas', 'overlay', 'positioning']):
- `sections` (array of string):
- `positions_mm` (array of number):
- `channels` (array of string):
- `atlas_layers` (array of string):
- `atlas_opacity` (number):
- `warp` (string, one of ['applied', 'none']):
- `layers` (array of string):
- `depth` (integer):
- `masks` (array of string):

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
        

- `box` (array of number):
- `picture` (integer):
- `region` (string):

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
            view: True returns the pictures described below; false returns none.

        Returns:
            The settings or recipe now in force, the channels there are, and for
            a derived channel, up to 4 of the sections (spread over the stack when
            none are named) two pictures each: the channel before this call, then
            after it; a channel this call creates has no before, so one picture each.
        

- `name` (string, required):
- `sections` (array of string):
- `weights` (object):
- `clahe_clip` (number):
- `clahe_tiles` (integer):
- `n4` (boolean):
- `denoise` (boolean):
- `contrast_limits` (array of number):
- `gamma` (number):
- `colormap` (string):
- `invert` (boolean):
- `reset` (boolean):
- `view` (boolean):

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
        

- `query` (string):
- `section` (string):
- `points` (array of any):

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
        

- `regions` (array of string, required):
- `positions_mm` (array of number, required):

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
        

- `path` (string):
- `pattern` (string):

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
        

- `query` (string, required):
- `path` (string):
- `glob` (string):

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
        

- `path` (string, required):
- `offset` (integer):
- `limit` (integer):

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
        

- `sections` (array of any):
- `cutting_angles` (any):
- `view` (boolean):

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
        

- `section` (string, required):
- `regions` (array of string, required):
- `note` (string):

## note

Append one line to the run notes, which are saved with the results.

        Args:
            text: The note.
        

- `text` (string, required):

## undo

Undo the last write. A tool call, or a finished piece of background work, is one step.

        Args:
            steps: How many steps to undo at once, newest first (all the history
                holds when it holds fewer).

        Returns:
            The sections undone and the fields that changed, and the rows of
            those sections.
        

- `steps` (integer):

## redo

Redo the write that ``undo`` reversed.

        Args:
            steps: How many undone steps to redo at once.

        Returns:
            The sections redone and the fields that changed, and the rows of
            those sections.
        

- `steps` (integer):

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
        

- `summary` (string, required):
- `notes` (array of string, required):
- `interval_breaks` (array of string, required):

## show_stack

Read an opening-picture page (1-based); read every page before writes.

- `page` (integer, required):
