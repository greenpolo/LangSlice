# QUINT

LangSlice writes each registration in the formats of the
[QUINT workflow](https://quint-workflow.readthedocs.io), so its tools open it
as is: [PyNutil](https://github.com/Neural-Systems-at-UIO/PyNutil) to count
cells per region, and VisuAlign to look at or adjust the warps.

## The files

After a run, `langslice/exports/` in your image folder holds:

| File | What it is |
|---|---|
| `visualign.json` | Each section's placement and warp. Use this one. |
| `quicknii.json` | The placement only, without the warps |
| `<name>_sNNN.png` | One copy of each section, numbered; the two files name these |

QuickNII and VisuAlign open the copies from the folder that holds the file,
so keep them together. They know the Allen mouse and Waxholm rat atlases
only, and run on Windows and macOS; PyNutil takes every BrainGlobe atlas and
runs anywhere.

## Count cells with PyNutil (recommended)

1. **Segment** your sections with any tool (ilastik, Cellpose, a threshold in
   Fiji): one image per section, the objects in one colour on a plain
   background. Name each after its section's copy, keeping the `_sNNN`:
   `M04_s007_seg.png` for `M04_s007.png`. Any size with the copy's
   proportions works.
2. **Install** PyNutil: `pip install PyNutil`, or the Windows and macOS app
   from its [releases](https://github.com/Neural-Systems-at-UIO/PyNutil/releases).
3. **Run** it. In the app, choose the atlas, `exports/visualign.json` as the
   registration JSON, your segmentation folder, and leave the object colour on
   **auto**. In Python:

   ```python
   from brainglobe_atlasapi import BrainGlobeAtlas
   import PyNutil as pnt

   atlas = BrainGlobeAtlas("allen_mouse_25um")
   alignment = pnt.read_alignment("/data/brains/M04/langslice/exports/visualign.json")
   segmentations = pnt.read_segmentation_dir("/data/brains/M04/segmentations", pixel_id="auto")
   coords = pnt.seg_to_coords(segmentations, alignment, atlas)
   counts = pnt.quantify_coords(coords, atlas)
   pnt.save_analysis("/data/brains/M04/pynutil", coords, atlas, label_df=counts)
   ```

PyNutil writes the object count and area of every region, per hemisphere,
and point clouds you can open in
[MeshView](https://meshview.apps.ebrains.eu/?atlas=ABA_Mouse_CCFv3_2017_25um).

- **The atlas:** `allen_mouse_25um` for every Allen mouse job, whichever
  resolution it was registered on: the export is on that atlas's grid. Any
  other job: its own atlas.
- **The colour:** leave it on auto, or draw the objects in black. PyNutil
  reads a colour you set as blue, green, red, so a red set as red finds
  nothing.
- If `BrainGlobeAtlas` fails with “cannot safely cast non-equivalent int64 to
  uint16”, run `pip install "pandas==3.0.5"`.

## Look at the warps in VisuAlign

Open `exports/visualign.json` in
[VisuAlign](https://quint-workflow.readthedocs.io/en/latest/VisuAlign.html).
Each section shows the atlas over it, warped as LangSlice fitted it. The
warp arrives as a dense grid of markers, about a thousand per section. Save
your changes with **File ‣ Save As** to a new file and give that file to
PyNutil. [VERSO](verso.md) opens the same file.
