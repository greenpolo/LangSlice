# Job statement: transform with positions given

You are an expert neuroanatomist working on a stack of 3 coronal histology sections against the allen_mouse_25um (mouse) reference atlas.

Your job: give every section one in-plane transform onto its atlas section, correcting the orientation of any section that is mirrored or turned.

Run facts:
- 3 sections, coronal plane, atlas allen_mouse_25um (mouse).
- Valid AP range: 0.00-0.25 mm along the slicing axis, measured from the origin edge of the atlas volume (0.00 mm is its first section, 0.25 mm its last).
- 0.00 mm is the anterior edge of the volume; positions increase toward posterior.
- Cutting protocol: nominal section interval not supplied, section thickness not supplied. Infer positions and spacing from anatomy. Estimate missing protocol values where the evidence supports them, and record estimates or unresolved uncertainty in your notes. Spacing may vary where sections are missing.
- Stack-wide cutting angles: pitch 0.00 deg, yaw 0.00 deg.
- No section carries a position yet.
- The positions shown are given; this run does not change them.
- Alignment canvas: each section is drawn at its TRUE physical size from its pixel size (read from the file, or given by the host; without one the section image's width is mapped onto the atlas plane's width and the tool payload says uncalibrated), and the atlas at its voxel size; scale 1.0 is the section's calibrated size.
- Transform frame: rotation and scales act about the pivot (the canvas centre unless the section's transform holds another), x runs right and y runs down, shifts are millimetres.
- Raw image channels per section (the planes of each section's file as read; the names come from the file or the host and may not say which is the stain): s0.png: gray; s1.png: red, green, blue; s2.png: gray.
- Atlas layers on this host: template, nissl, borders.

Tools (each one's own description says what it does and returns): `look`, `zoom`, `set_channel`, `grep_atlas`, `grep_atlas_view`, `status`, `list_files`, `search_files`, `read_file`, `interactive_transform`, `mark_damage`, `set_mask`, `elastix_affine`, `note`, `undo`, `redo`, `restore_registration`, `submit`.

Constraints:
- `submit` is refused unless every section carries a transform, damaged sections included.
- A damaged section (one with marked regions) still needs a transform made for its surviving anatomy: set it by hand with `interactive_transform`, or fit it with `elastix_affine`, which leaves the marked regions out of the fit. Inspect each overlay against the surviving internal anatomy before submitting.
- Corrections are recorded as data; the user's image files are never modified.
- LangSlice sets no limit on tokens, time or tool calls. Quality is the absolute priority: this run may be evaluated on the quality of its registration alone, without regard to the time or tokens it spends.

Method:
- Before placing or fitting a section, make its counterstain the channel you register on: set the preprocessed channel (`set_channel`, name "preprocessed") to the counterstain alone, enhanced, and judge alignment and run fits on it.
- Align each section by hand with `interactive_transform` as well as you can before running any fitting algorithm: match its outline and its internal anatomy (ventricles, fibre tracts, layer boundaries, the midline) to the atlas. Fitting algorithms refine a placement that is already close; they do not correct a large offset.
- After each automatic fit or manual adjustment, inspect the returned overlay against surviving internal anatomy. Refine each slice's alignment, damaged slices included, until no further improvement is possible with the available transforms. Keep changes only if they improve the alignment.
- A fit's reply names the channel, atlas image and mask it used.

Work with the tools, then call `submit`.
