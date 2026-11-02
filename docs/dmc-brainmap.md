# DMC-BrainMap

[DMC-BrainMap](https://github.com/hejDMC/napari-dmc-brainmap)
([docs](https://napari-dmc-brainmap.readthedocs.io/en/latest/),
[paper](https://doi.org/10.1016/j.crmeth.2026.101302)) is a napari plugin
that maps cells to an atlas. LangSlice writes a DMC-BrainMap animal folder,
so DMC-BrainMap counts cells on LangSlice's registration instead of its own.

:::{admonition} Linear only
:class: warning

DMC-BrainMap holds a linear registration: position, cutting angles and
in-plane alignment. Warps fitted in LangSlice are left out, so cells are
placed where the section sat before the warp. If your sections are warped,
count with [PyNutil](quint.md) on `visualign.json` instead.
:::

## Export

With LangSlice's Python:

```python
import langslice

job = langslice.open_job("/data/brains/M04")
job.export_dmc_brainmap()
```

This writes `langslice/exports/dmc_brainmap/M04/`, named after your image
folder, which DMC-BrainMap reads as the animal id:

| File | What it is |
|---|---|
| `params.json` | The atlas and cutting plane, in DMC-BrainMap's settings format |
| `sharpy_track/dapi/` | Each section resized to the atlas plane (`<name>_downsampled.tif`), and `registration.json` |
| `langslice_export.json` | Sections written without a registration (`not_registered`) and sections whose warp was left out (`warps_left_out`) |

The registration goes in the `dapi` channel folder; pass `channel="green"`
(or `n3`, `cy3`, `cy5`) for another. Export again after any correction.

## Count cells in DMC-BrainMap

1. **Install** DMC-BrainMap 0.2.0b2 in its own environment:

   ```bash
   conda create -n dmc-brainmap python=3.12 -y
   conda activate dmc-brainmap
   pip install "napari-dmc-brainmap==0.2.0b2" pyqt5
   napari
   ```

2. **Look** at the registration: **Plugins ‣ dmc_brainmap ‣ Registration**,
   input path the animal folder, registration channel **dapi**, **Start
   Registration GUI**. Answer **Yes** to loading the registration it finds;
   **No** overwrites it.
3. **Add your images** to the animal folder as `rgb/<name>_rgb.tif`, one per
   section, `<name>` as in `sharpy_track/dapi/<name>_downsampled.tif`. Any
   size with the proportions of the images LangSlice registered.
4. **Segment** them in the **Segmentation** widget, or write
   `segmentation/cells/<channel>/<name>_cells.csv` yourself, with
   `Position X` and `Position Y` in pixels of the `rgb` image.
5. **Count:** **Plugins ‣ dmc_brainmap ‣ Create results file**, input path
   the animal folder, segmentation folder **rgb**, registration channel
   **dapi**, segmentation type **cells**. Each cell's region and atlas
   coordinates go to `results/cells/<channel>/M04_cells.csv`.

## Limits

- Coronal and sagittal sections. Sagittal results need 0.2.0b2; the stable
  0.1.8 fails on them.
- DMC-BrainMap stores positions in whole atlas voxels, so a cell can move up
  to one voxel from LangSlice's placement. That changes its region only at a
  border.
- In the registration window, sections look stretched, because they are
  resized to the atlas plane, and sagittal planes show anterior at the top.
- DMC-BrainMap knows bregma for `allen_mouse_10um`, `whs_sd_rat_39um` and
  `azba_zfish_4um` only. On other atlases it estimates bregma, which shifts
  the millimetre coordinates in its tables but not the regions.
