# Export results for cell counting

LangSlice registers sections. It does not detect or count cells. This page
shows which files it writes so that you can assign your cells to atlas
regions in other software.

## What you have after a run

When a registration run submits, the job folder holds three result files:

| File | What it is | Open it with |
|---|---|---|
| `registration.json` | Each section's placement in the atlas, in a readable form | A text editor, or a script |
| `exports/quicknii.json` | Each section's linear placement (position, angles, in-plane alignment), without shear | QuickNII, ABBA |
| `exports/visualign.json` | The same, plus VisuAlign markers that carry each section's warp and shear | VisuAlign, Nutil, PyNutil |

The job folder is the `langslice` folder inside your image folder.

## Write the region maps

For counting, you usually want a map of atlas regions for each section, one
value per pixel. LangSlice writes these maps only when asked, because they
are large. Ask your agent:

```text
Write the region maps for /data/brains/M04.
```

For maps at the full resolution of your image files, say so:

```text
Write the region maps for /data/brains/M04 at full resolution.
```

Each section gets these files in `sections/<name>/` inside the job folder,
where `<name>` is the filename without its ending:

| File | What it holds | Use it for |
|---|---|---|
| `labels_fiji.tif` | Each pixel's atlas region as a small number, coloured with the atlas colours | Looking at the regions in Fiji |
| `labels.csv` | The table from those numbers to atlas id, acronym, name and colour | Naming the regions in `labels_fiji.tif` |
| `labels.tif` | Each pixel's atlas region id, 0 outside the atlas | Counting cells per region with a script |
| `coords.tif` | Each pixel's atlas coordinates in micrometres, three channels | Placing cells in atlas space |
| `residual.tif` | The warp, on sections that have one | Rarely needed directly: `coords.tif` already includes it |
| `maps.json` | The map size, the micrometres per map pixel and how it relates to the image | Converting between map and image pixels |

Writing the maps also rewrites the three result files from the job as it
stands.

:::{note}
By default the maps are the size of LangSlice's working copy of each
section, at most 3072 pixels on the long side. Cell positions measured on
the full-size image must be scaled to the map's size first, or ask for maps
at full resolution. Full-resolution maps of a whole-slide scan take hundreds
of megabytes per section.
:::

## Count cells per region

1. Detect cells on your section images in your usual software, such as
   QuPath, Fiji or CellProfiler. Export each cell's x and y in image pixels.
2. For each cell, read the region in `labels.tif` at the cell's pixel,
   scaled to the map's size when the map is not at full resolution.
3. Name each region with `labels.csv`, or with the atlas's own list of
   regions.
4. Add up the cells per region.

Your agent can write this script for you. Give it the job folder and the
file of cell positions, and say which columns hold x and y.

To look at the regions on a section, open `labels_fiji.tif` in Fiji. Each
region shows in its atlas colour. The pixel value under the cursor is the
`index` in `labels.csv`, whose row names the region.

<details>
<summary>Under the hood: the coordinate conventions</summary>

Map pixels are `[row, col]`, row 0 at the top, pixel centres at whole
numbers, in the image file's own orientation (no rotation or mirroring
applied). `coords.tif` gives atlas micrometres in the atlas's own axis
order. For the Allen mouse atlas (orientation `asr`), the channels run
anterior to posterior, superior to inferior, and right to left. Outside the
atlas they hold NaN. `labels.tif` stores the atlas's own region ids as 32-bit
numbers, which Fiji cannot open without loss; that is why `labels_fiji.tif`
exists. The full formats are in
[Job folder and file formats](../reference/file_formats.md).

</details>

## Use QuickNII, VisuAlign, Nutil, PyNutil or ABBA

- **QuickNII and VisuAlign.** `quicknii.json` and `visualign.json` are in
  the QUINT JSON format (a plain-text data format) that QuickNII, VisuAlign
  and DeepSlice write. Beside them, the `exports` folder holds a PNG copy
  of each section, named with its section number, for example
  `M04_B_07_s013.png`, and the files name these copies. QuickNII and
  VisuAlign read PNG and JPEG only and look for the images next to the
  file, so open the files where they are. Each copy is the gray image the
  agent looked at, at LangSlice's working size (at most 3072 pixels on the
  long side). For the Allen mouse atlas, at any resolution, the target is
  `ABA_Mouse_CCFv3_2017_25um.cutlas`.
- **Shear.** LangSlice's in-plane alignment can shear a section a little
  (its two image axes not quite at right angles). QuickNII and VisuAlign
  cannot show shear: they straighten the axes when they open a file. So
  both files hold the nearest placement without shear, which every program
  reads the same way, and the markers in `visualign.json` add back the
  shear together with the warp. `visualign.json` therefore reproduces
  LangSlice's placement in VisuAlign, Nutil and PyNutil.
  `quicknii.json` alone lacks the shear: on two sections sheared by 3 and
  4 degrees, it moved the image corners by about 200 µm from where
  LangSlice put them, less toward the middle. Use `visualign.json` for
  counting.
- **How close the markers come.** VisuAlign draws a warp as straight
  pieces between markers, about 900 per section. A FireANTs warp bends on
  a finer scale, so between markers the file follows it approximately:
  typically within 4 to 8 µm, and within 18 to 36 µm for 95% of the tissue,
  on the sections we measured. `coords.tif` and `labels.tif` are exact.
- **Nutil and PyNutil.** From VisuAlign, the QUINT workflow (Nutil) counts
  objects per region; PyNutil reads `visualign.json` directly. PyNutil
  ignores the atlas named in the file: it places the sections in whatever
  atlas you load. LangSlice writes every Allen mouse job on the 25 µm grid,
  so load `allen_mouse_25um` in PyNutil, also for jobs on
  `allen_mouse_10um` or `ccfv3augmented_mouse_25um`. Another atlas is
  written on its own grid: load the job's atlas. PyNutil triangulates the
  markers itself, which takes about half a minute per section.
- **Name your segmentations by section number.** Nutil and PyNutil pair a
  segmentation image with a section by the `_sNNN` in its filename. Give
  each segmentation the same `_sNNN` as the section's copy, for example
  `M04_B_07_s013_seg.png` for `M04_B_07_s013.png`, with no other `_s`
  before it in the name: Nutil reads the first `_s` it finds. The number is
  the one already in your filenames when every image has a `_sNNN` of its
  own; otherwise it is the section's place in LangSlice's order, counting
  from 1. For PyNutil's table of cell coordinates, put that number in the
  `section number` column. A segmentation can have any resolution, as long
  as it keeps the image's proportions. A section paired with the wrong
  number is counted with another section's registration, and no program
  warns.
- **ABBA.** ABBA imports the linear placement from `quicknii.json` with
  **ABBA - Import QuickNII Project**. ABBA's own dialog warns that this import
  is not exact. ABBA has no VisuAlign import, so the warps do not come
  across.

Other atlases export to QuickNII targets that assume the BrainGlobe voxel
grid; that is untested. The rat target is `WHS_Rat_v4_39um.cutlas`.

## After a correction

A correction changes the registration but not maps written earlier.
`registration.json` marks each section's maps `current: false` once the
section has changed. Ask for the maps again after any correction.
