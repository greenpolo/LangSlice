# Register a brain

You register a brain by asking your AI app. Its agent places each section in
the atlas with LangSlice's [tools](tools.md), on your computer, and checks the
result with you.

## Put the sections in one folder

- One folder per brain, one image file per section: TIFF (including OME-TIFF
  and whole-slide pyramids), PNG or JPEG. Every image in the folder is read as
  a section, so leave overviews and slide labels out.
- Export microscope formats (`.czi`, `.vsi`, `.ndpi`, `.svs`) to TIFF first,
  for example in Fiji with **File ‣ Save As ‣ Tiff…**.
- Keep a channel that shows the anatomy (DAPI, Nissl, NeuN). A tracer channel
  alone shows too little of it.

LangSlice never changes your images. It writes its work into a folder named
`langslice` inside the image folder.

## Ask your agent

Start a new conversation in your AI app, one per brain, and say what to
register and what you know:

```text
Register the brain in /data/brains/M04. Coronal sections, Allen mouse atlas. The images are 0.65 µm per pixel. The sections are 40 µm thick and every sixth one was kept. M04_s07.tif has a torn left cortex. Never mirror a section.
```

Give the pixel size when the files don't record it: PNG and JPEG never do.
Without it, LangSlice sizes each section to the width of the atlas. Name
sections by filename. What you say must not change (“keep my positions”) is
locked: every tool refuses to change it.

To watch the run, ask your agent to open the LangSlice window on the folder.
It shows each picture the agent looks at and every tool call. Closing it does
not stop the run.

## Check the result

The agent can be wrong without noticing. Look at every section's overlay, the
section with the atlas borders drawn on it: the borders should follow the
anatomy inside the tissue, such as the ventricles, the corpus callosum and the
hippocampus, not only the outline. On the positioning picture, the sections
should sit in cutting order, each at the atlas level with the same structures.

Tell your agent what is wrong, by filename: “M04_s07.tif is placed too far
anterior.” It runs again from where the job stands. Every change is a step you
can undo (“undo the last two steps”), and each section's earlier registrations
are kept (“put M04_s07.tif back to registration 2”).

## Get the results

The `langslice` folder holds, once the run ends:

| File | What it is |
|---|---|
| `registration.json` | Each section's placement in the atlas |
| `exports/quicknii.json`, `exports/visualign.json` | For the [QUINT](quint.md) tools (PyNutil, VisuAlign), with a PNG copy of each section beside them. `visualign.json` carries the warps. |

Region maps are written when you ask (“write the region maps for
/data/brains/M04”), into `sections/<name>/`: `labels.tif` holds each pixel's
atlas region id, `labels_fiji.tif` and `labels.csv` show the regions in Fiji,
and `coords.tif` holds each pixel's atlas coordinates. To count cells per
region, use [PyNutil](quint.md), or read `labels.tif` at each cell's pixel. The maps are at LangSlice's working size, at most 3072 pixels across,
unless you ask for full resolution. Ask for them again after a correction.

For DMC-BrainMap, ask your agent to export the brain for DMC-BrainMap. That
export carries the linear placement only, without the warps.

## Start from an existing registration

If you already placed the sections in QuickNII, VisuAlign, DeepSlice or
[ABBA](abba.md), give your agent the file: “Start from my QuickNII file
/data/brains/M04/series.json and add a warp to each section.” LangSlice keeps
your placement and fits a warp to each section. Say so if the agent may also
correct the placement.
