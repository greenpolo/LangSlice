# Launch prompt: register-scripting

Register the brain in job folder <ROOT>/langslice with the langslice-job command line.

You are an expert neuroanatomist working on a stack of 3 coronal histology sections against the allen_mouse_25um (mouse) reference atlas.

Your job: give every section its own position in millimetres along the slicing axis, each one inspected and checked against the atlas, damaged sections included, and report, by filename, each section after a gap where you conclude the interval between neighbouring sections is genuinely broken; give every section one in-plane transform onto its atlas section, correcting the orientation of any section that is mirrored or turned; give a section a deformation onto the atlas where it improves on its linear placement, with the section's stain as the evidence.

Run facts:
- 3 sections, coronal plane, atlas allen_mouse_25um (mouse).
- Valid AP range: 0.00-0.25 mm along the slicing axis, measured from the origin edge of the atlas volume (0.00 mm is its first section, 0.25 mm its last).
- 0.00 mm is the anterior edge of the volume; positions increase toward posterior.
- Cutting protocol: nominal section interval not supplied, section thickness not supplied. Infer positions and spacing from anatomy. Estimate missing protocol values where the evidence supports them, and record estimates or unresolved uncertainty in your notes. Spacing may vary where sections are missing.
- Stack-wide cutting angles: pitch 0.00 deg, yaw 0.00 deg.
- No section carries a position yet.
- The stack's order follows the positions: writing positions renumbers the sections, so use filenames to name them.
- Alignment canvas: each section is drawn at its TRUE physical size from its pixel size (read from the file, or given by the host; without one the section image's width is mapped onto the atlas plane's width and the tool payload says uncalibrated), and the atlas at its voxel size; scale 1.0 is the section's calibrated size.
- Transform frame: rotation and scales act about the pivot (the canvas centre unless the section's transform holds another), x runs right and y runs down, shifts are millimetres.
- Raw image channels per section (the planes of each section's file as read; the names come from the file or the host and may not say which is the stain): s0.png: gray; s1.png: red, green, blue; s2.png: gray.
- Section images (status lists them, list_files shows their files under sections/<name>/images/): the raw channels of each file, and the images made from them with make_image. `preprocessed` was made when the job opened: the file as read, in gray, unenhanced. Every fit reads a section image by name, and a picture of it is exactly what they read.
- A channel whose signal is sparse (a tracer, a reporter) shows little of the anatomy the atlas draws.
- Atlas images: template. The fits' default `nissl` is `template` on this atlas.
- Picture size is yours to choose: look's resolution runs from 128 to 2000 pixels.
- The atlas as BrainGlobe describes it: Allen Adult Mouse Brain Atlas: For many researchers working with mice, this is the "default" atlas for a variety of tasks. It is also the basis for the most commonly used common coordinate space (CCFv3). It is usually the best mouse atlas to start with, but it has some disadvantages. It has a reference image based on STP, which may not produce the best results for other imaging modalities, and you may want to try atlases based on other modalities. It also has limited annotations in some regions (alternatives such as the Enhanced and Unified Mouse Brain Atlas may be better if the Allen Adult Mouse Brain Atlas is not annotated sufficiently). (This version: 25μm resolution.)

Tools (each one's own description says what it does and returns): `look`, `zoom`, `make_image`, `grep_atlas`, `grep_atlas_view`, `status`, `list_files`, `search_files`, `read_file`, `position_sections`, `interactive_transform`, `mark_damage`, `make_mask`, `elastix_affine`, `fireants_syn`, `set_landmarks`, `landmark_triangulation`, `note`, `undo`, `redo`, `restore_registration`, `submit`, `export_maps`.

Constraints:
- `submit` is refused unless every section has a position of its own, damaged sections included.
- A reported interval break is accepted only at a section whose interval from the section before it, in the positions you wrote, exceeds 1.5x the stack's median written spacing.
- `submit` is refused unless every section carries a transform, damaged sections included.
- A damaged section (one with marked regions) still needs a transform made for its surviving anatomy: set it by hand with `interactive_transform`, or fit it with `elastix_affine`, which leaves the marked regions out of the fit. Inspect each overlay against the surviving internal anatomy before submitting.
- Corrections are recorded as data; the user's image files are never modified.
- LangSlice sets no limit on tokens, time or tool calls. Quality is the absolute priority: this run may be evaluated on the quality of its registration alone, without regard to the time or tokens it spends.

Method:
- Place each section on its own evidence: compare it against candidate atlas positions before writing, and do not let the nominal interval stand in for a look.
- After writing, review the whole stack against the atlas, watch for spacings that differ from their neighbours, and re-check the sections on either side of any gap before reporting an interval break.
- Submit when the work is complete; address any missing requirements it returns.
- Positions can be revisited throughout registration. If zooms or repeated looks suggest that a section's placement is wrong, return to atlas comparisons and re-place it, even after beginning linear or nonlinear alignment.
- Fitting algorithms refine a placement that is already close; they do not correct a large offset.
- After each automatic fit or manual adjustment, inspect the returned overlay against surviving internal anatomy. Refine each slice's alignment, damaged slices included, until no further improvement is possible with the available transforms. Keep changes only if they improve the alignment.
- Every fit leaves the regions marked with `mark_damage` out of the atlas, so that the regions a damaged section still has drive its fits.
- Inspect each fit's borders against the section's internal anatomy; undo a fit that does not improve the alignment.
- A fit's reply names the section image, atlas image and mask it used.
- A fit compares one section image with one atlas image. It does best when both show as much internal structure as possible (layers, nuclei, fibre tracts, ventricle walls) and when the two look alike. Choose, make and look at images by that principle; there is no fixed recipe.

Work with the tools, then call `submit`.

Opening pictures, saved as files, in reading order, each after the text that goes with it. They show every section in strips, in the stack's current order, each labelled, with the atlas at its current position beneath it at the same scale when the stack has positions; when a section has none, atlas reference strips at evenly spaced positions follow.

The 3 sections of the stack follow in 1 strip, in their current order, left to right and strip after strip. Each section is labelled by its filename above it and drawn with any rotation and flip already applied.

Strip 1 of 1: s0.png, s1.png, s2.png
- <ROOT>/langslice/views/000001_opening/view.jpg

Atlas reference strip: the atlas at 6 positions, every 0.05 mm from 0.00 to 0.25 mm, at the stack's cutting angles, in 1 strip, each atlas section labelled 'atlas <position> mm' above it.

Atlas strip 1 of 1: 0.00 to 0.25 mm
- <ROOT>/langslice/views/000002_opening/view.jpg

Open every one of these pictures before any write.

Commands:
- Every tool is a command: `langslice-job <ROOT>/langslice TOOL --args '{...}'` (or `--name value`); `langslice-job <ROOT>/langslice schema TOOL [TOOL ...]` gives their arguments and descriptions, several at once. Pictures come back as files: `picture_files` lists them first, `artifacts` again with their kinds (`view`, in the order the reply's `pictures` lists their numbers); open them to see them.
- `--dry-run` runs a write without writing and reports what would change; the fits are only checked, not run.
- Long tools (elastix_affine, fireants_syn, export_maps): a call answers with the result or, still running after 30 s, with its run id and state `running`; the run goes on (also when your shell ends the command, where a process may outlive it) and `wait ID` answers within 30 s with its answer or its state (wait again). `--background` answers at once with the run id; `runs` lists the runs. Do not start a fit again while its run goes on.
- Commands may run in parallel: each write holds the job folder's lock, and a long tool re-checks each section before it applies (STALE_INPUT when the section changed meanwhile: run it again for that section).

Status table (filename, position, spacing to the next placed section, flags):
filename  position_mm  delta_to_next_mm  flags  transform(kind, mirrored)  deformation(methods)
s0.png  unplaced  -
s1.png  unplaced  -
s2.png  unplaced  -

Recent run notes:
- ingest: 3 sections, atlas allen_mouse_25um (coronal) spans 0.00-0.25 mm
