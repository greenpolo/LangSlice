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
- Section images (status lists them, list_files shows their files under sections/<name>/images/): the raw channels of each file, and the images made from them with make_image. `preprocessed` was made when the job opened: the file as read, in gray, unenhanced. Every fitreads a section image by name, and a picture of it is exactly what they read.
- A channel whose signal is sparse (a tracer, a reporter) shows little of the anatomy the atlas draws.
- Atlas images: template. The fits' default `nissl` is `template` on this atlas.
- The atlas as BrainGlobe describes it: Allen Adult Mouse Brain Atlas: For many researchers working with mice, this is the "default" atlas for a variety of tasks. It is also the basis for the most commonly used common coordinate space (CCFv3). It is usually the best mouse atlas to start with, but it has some disadvantages. It has a reference image based on STP, which may not produce the best results for other imaging modalities, and you may want to try atlases based on other modalities. It also has limited annotations in some regions (alternatives such as the Enhanced and Unified Mouse Brain Atlas may be better if the Allen Adult Mouse Brain Atlas is not annotated sufficiently). (This version: 25μm resolution.)

Tools (each one's own description says what it does and returns): `look`, `zoom`, `make_image`, `grep_atlas`, `grep_atlas_view`, `status`, `list_files`, `search_files`, `read_file`, `interactive_transform`, `mark_damage`, `make_mask`, `elastix_affine`, `note`, `undo`, `redo`, `restore_registration`, `submit`.

Constraints:
- `submit` is refused unless every section carries a transform, damaged sections included.
- A damaged section (one with marked regions) still needs a transform made for its surviving anatomy: set it by hand with `interactive_transform`, or fit it with `elastix_affine`, which leaves the marked regions out of the fit. Inspect each overlay against the surviving internal anatomy before submitting.
- Corrections are recorded as data; the user's image files are never modified.
- LangSlice sets no limit on tokens, time or tool calls. Quality is the absolute priority: this run may be evaluated on the quality of its registration alone, without regard to the time or tokens it spends.

Method:
- Fitting algorithms refine a placement that is already close; they do not correct a large offset.
- After each automatic fit or manual adjustment, inspect the returned overlay against surviving internal anatomy. Refine each slice's alignment, damaged slices included, until no further improvement is possible with the available transforms. Keep changes only if they improve the alignment.
- A fit's reply names the section image, atlas image and mask it used.
- A fit compares one section image with one atlas image. It does best when both show as much internal structure as possible (layers, nuclei, fibre tracts, ventricle walls) and when the two look alike. Choose, make and look at images by that principle; there is no fixed recipe.

Work with the tools, then call `submit`.
