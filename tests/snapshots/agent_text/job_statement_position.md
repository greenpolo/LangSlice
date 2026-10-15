# Job statement: positioning only

You are an expert neuroanatomist working on a stack of 3 coronal histology sections against the allen_mouse_25um (mouse) reference atlas.

Your job: give every section its own position in millimetres along the slicing axis, each one inspected and checked against the atlas, damaged sections included, and report, by filename, each section after a gap where you conclude the interval between neighbouring sections is genuinely broken.

Run facts:
- 3 sections, coronal plane, atlas allen_mouse_25um (mouse).
- Valid AP range: 0.00-0.25 mm along the slicing axis, measured from the origin edge of the atlas volume (0.00 mm is its first section, 0.25 mm its last).
- 0.00 mm is the anterior edge of the volume; positions increase toward posterior.
- Cutting protocol: nominal section interval not supplied, section thickness not supplied. Infer positions and spacing from anatomy. Estimate missing protocol values where the evidence supports them, and record estimates or unresolved uncertainty in your notes. Spacing may vary where sections are missing.
- Stack-wide cutting angles: pitch 0.00 deg, yaw 0.00 deg.
- No section carries a position yet.
- The stack's order follows the positions: writing positions renumbers the sections, so use filenames to name them.
- The orientation shown is fixed; transforms are not part of this run.
- Raw image channels per section (the planes of each section's file as read; the names come from the file or the host and may not say which is the stain): s0.png: gray; s1.png: red, green, blue; s2.png: gray.
- Atlas layers on this host: template, borders.

Tools (each one's own description says what it does and returns): `look`, `zoom`, `set_channel_properties`, `set_preprocessed_channel_properties`, `grep_atlas`, `grep_atlas_view`, `status`, `list_files`, `search_files`, `read_file`, `position_sections`, `mark_damage`, `note`, `undo`, `redo`, `submit`.

Constraints:
- `submit` is refused unless every section has a position of its own, damaged sections included.
- A reported interval break is accepted only at a section whose interval from the section before it, in the positions you wrote, exceeds 1.5x the stack's median written spacing.
- Corrections are recorded as data; the user's image files are never modified.
- LangSlice sets no limit on tokens, time or tool calls. Quality is the absolute priority: this run may be evaluated on the quality of its registration alone, without regard to the time or tokens it spends.

Method:
- Before placing or fitting a section, make its counterstain the channel you register on: set the preprocessed channel (`set_preprocessed_channel_properties`) to the counterstain alone, enhanced, and judge alignment and run fits on it.
- Place each section on its own evidence: compare it against candidate atlas positions before writing, and do not let the nominal interval stand in for a look.
- After writing, review the whole stack against the atlas, watch for spacings that differ from their neighbours, and re-check the sections on either side of any gap before reporting an interval break.
- Submit when the work is complete; address any missing requirements it returns.
- Positions can be revisited throughout registration. If zooms or repeated looks suggest that a section's placement is wrong, return to atlas comparisons and re-place it, even after beginning linear or nonlinear alignment.

Work with the tools, then call `submit`.
