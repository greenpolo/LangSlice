# ABBA

A registration made in LangSlice opens in ABBA as a project of its own.

1. In the LangSlice window choose **Export ‣ ABBA project (.abba)**. From
   Python, `job.export_abba()` writes the same file.
2. In ABBA (Allen Brain Adult Mouse Brain CCF 2017, coronal) choose
   **File ‣ Load State** and pick
   `<images>/langslice/exports/<images>.abba`.

Each section opens from its own image file with its alignment and warp; edit
a warp with **Edit last registration**. The project uses ABBA's own
registration types, so it saves and reopens without LangSlice.

It works with ABBA 0.25 (0.24.1 too), the Allen Mouse V3 or V3p1 atlas, and
coronal sections.

## Limits

- One cutting angle for the whole stack, as in ABBA.
- A section left out of the file is named, with the reason, in
  `exports/abba_export.json`.
