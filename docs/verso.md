# VERSO

:::{admonition} VERSO is in early development
:class: warning

VERSO has no release yet. Its features and file formats can change without
notice.
:::

[VERSO](https://github.com/LeonardoLupori/verso) is a desktop app for
registering brain sections to the Allen mouse atlas by hand: you place the
atlas on each section, then bend it with control points. It opens
LangSlice's `visualign.json`, so you can check and adjust a LangSlice
registration there, then count cells with PyNutil.

It works with LangSlice jobs on the Allen mouse atlas, coronal or sagittal.
It does not open Waxholm rat jobs correctly.

## Install

VERSO installs from source with [uv](https://docs.astral.sh/uv/), which
also fetches the Python 3.12 that VERSO needs:

```bash
git clone https://github.com/LeonardoLupori/verso.git
cd verso
uv sync
uv run verso
```

The first time you open a project, VERSO downloads the Allen mouse atlas
through BrainGlobe, so it needs an internet connection.

## Open a LangSlice registration

1. Choose **File ‣ Import ‣ QuickNII / VisuAlign project…**.
2. Give the project a **Name** and a **Location**. VERSO creates a project
   folder there.
3. Under **Alignment file**, click **Browse…** and choose
   `langslice/exports/visualign.json` in your image folder. VERSO fills in
   the atlas, the cutting plane and the **Registration images** (LangSlice's
   copies, from the same folder).
4. Choose which images VERSO works on:
   - To refine and count with PyNutil, leave **Use these images as the
     full-resolution originals** ticked. VERSO uses the copies.
   - For VERSO's own full-resolution exports, untick it, click **Add
     images…** and choose your original image files. VERSO matches each to
     its section by file name.

   Both give the same registration.
5. Click **OK**.

Open the **Warp** view to see each section with the atlas warped as
LangSlice fitted it.

## Refine, then count with PyNutil

1. Adjust sections in the **Align** and **Warp** views, and save with
   **File ‣ Save all** (Ctrl+S).
2. Choose **Export ‣ For QUINT ‣ VisuAlign JSON…** and save the file.
   VERSO offers to write a PNG copy of each section beside it; QuickNII and
   VisuAlign need these, PyNutil does not.
3. Count with PyNutil as in [QUINT](quint.md), giving it the file VERSO
   wrote instead of LangSlice's `visualign.json`. The export keeps each
   section's number, so segmentations named with `_sNNN` still match.

## Limits

- LangSlice's warp arrives as a dense grid of about 1,000 control points per
  section. In the **Warp** view you move them one at a time.
- Count with PyNutil rather than with VERSO's **Export ‣ Quantify**. VERSO
  reads each pixel's region from the opposite hemisphere of the atlas, so at
  borders where the atlas's two halves differ, its regions differ from
  PyNutil's (about 2% of tissue pixels, all near borders).
