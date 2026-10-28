# Export results for cell counting

LangSlice registers sections. It does not detect or count cells. This page
shows which files it writes so that you can assign your cells to atlas
regions in other software.

## What you have after a run

When a registration run submits, the job folder holds three result files:

| File | What it is | Open it with |
|---|---|---|
| `registration.json` | Each section's placement in the atlas, in a readable form | A text editor, or a script |
| `exports/quicknii.json` | Each section's linear placement (position, angles, in-plane alignment) | QuickNII, ABBA |
| `exports/visualign.json` | The same, plus each section's warp as VisuAlign markers | VisuAlign, then the QUINT workflow (Nutil) |

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

## Use QuickNII, VisuAlign or ABBA

- **QuickNII and VisuAlign.** `quicknii.json` and `visualign.json` are in
  the QUINT JSON format (a plain-text data format) that QuickNII, VisuAlign and DeepSlice write. They
  name your image files as they are. ABBA's QuickNII export saves JPEG
  images because, in its own words, that is "the format QuickNII reads".
  Sections in TIFF files may need JPEG or PNG copies. For the Allen mouse atlas, at any
  resolution, the target is `ABA_Mouse_CCFv3_2017_25um.cutlas`. The markers
  in `visualign.json` carry each section's warp, so VisuAlign shows the
  sections as LangSlice left them. From VisuAlign, the QUINT workflow
  (Nutil) counts objects per region.
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
