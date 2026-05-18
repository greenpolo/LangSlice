"""Bake all static atlas assets the GH Pages SPA needs from BrainGlobe's
``allen_mouse_25um`` into ``web-demo/public/atlas/allen_mouse_25um/``.

Outputs (528 sections at native 25 um step = 0.025 mm AP):
  manifest.json          atlas metadata + section index
  structures.json        id -> {acronym, name, rgb_triplet, parent_path}
  sections/NNN.png       Nissl reference slice, RGBA w/ root-mask alpha
  colored/NNN.png        colored-regions slice (per rgb_triplet), RGBA w/ alpha
  borders/NNN.png        per-slice region borders, 1-bit alpha
  mesh/root.obj          root brain mesh (id 997 for Allen mouse)

NNN is the AP voxel index (zero-padded, 000..527). The manifest provides
the index -> mm mapping using the harness's existing
``index_to_position_mm`` so the browser side stays in atlas-mm coords.

Run:
    python web-demo/scripts/build_atlas_bundle.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from brainglobe_atlasapi import BrainGlobeAtlas

# Reuse the existing atlas helpers so we render exactly what the rest of the
# codebase already trusts (including image_gen registration's colored regions).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from langslice_harness.atlas import (  # noqa: E402
    get_reference_slice,
    index_to_position_mm,
)
from langslice_harness.atlas.space import (  # noqa: E402
    Plane,
    atlas_space_context,
    slice_axis_index,
)
from langslice_harness.atlas.core import orient_slice_for_display  # noqa: E402

ATLAS_NAME = "allen_mouse_25um"
PLANE: Plane = "coronal"
OUT_ROOT = Path(__file__).resolve().parents[1] / "public" / "atlas" / ATLAS_NAME

# Reasonable subset of structures whose IDs are deliberately suppressed in the
# colored-region rendering (matches image_gen_helpers ventricle suppression).
_VENTRICLE_KEYWORDS = {"ventricle", "central canal", "choroid", "subependymal"}


def _is_ventricle(structure: dict) -> bool:
    name = str(structure.get("name", "")).lower()
    return any(kw in name for kw in _VENTRICLE_KEYWORDS)


def _root_mask_for_slice(annotation_2d: np.ndarray) -> np.ndarray:
    """Binary mask: pixel is in the brain (annotation != 0)."""
    return (annotation_2d != 0).astype(np.uint8) * 255


def _coloring_palette(atlas: BrainGlobeAtlas) -> dict[int, tuple[int, int, int]]:
    """Map structure id -> RGB triplet, suppressing ventricles."""
    palette: dict[int, tuple[int, int, int]] = {}
    for sid, struct in atlas.structures.items():
        if _is_ventricle(struct):
            continue
        triplet = struct.get("rgb_triplet")
        if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
            palette[int(sid)] = (int(triplet[0]), int(triplet[1]), int(triplet[2]))
    return palette


def _colored_slice_rgba(
    annotation_2d: np.ndarray,
    palette: dict[int, tuple[int, int, int]],
) -> Image.Image:
    """Render a coronal annotation slice as a colored-regions RGBA image.

    Background (annotation==0) is transparent; ventricles fall back to gray;
    everything else gets its rgb_triplet."""
    h, w = annotation_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for uid in np.unique(annotation_2d):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        color = palette.get(uid_int, (128, 128, 128))
        mask = annotation_2d == uid_int
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def _border_slice(annotation_2d: np.ndarray) -> Image.Image:
    """Per-slice region borders: pixels where any 4-neighbor has a different
    region id (including the brain↔background transition, so the outermost
    cortical / cerebellar / OB silhouette stays visible). Encoded as RGBA
    with alpha=255 on borders, 0 elsewhere so it composes on top of the
    colored slice in the 3D viewer."""
    a = annotation_2d.astype(np.int64)
    diff_h = np.zeros_like(a, dtype=bool)
    diff_v = np.zeros_like(a, dtype=bool)
    diff_h[:, 1:] = a[:, 1:] != a[:, :-1]
    diff_v[1:, :] = a[1:, :] != a[:-1, :]
    # Mark a pixel as a border whenever its left/up neighbor has a different
    # id. We intentionally don't mask by `a != 0` so brain↔background edges
    # register — that draws the brain silhouette. Pure background regions
    # have a == 0 == neighbor, so diff_h | diff_v is False there → alpha 0.
    border = diff_h | diff_v
    rgba = np.zeros((*a.shape, 4), dtype=np.uint8)
    # Cyan to match the desktop registration overlay convention.
    rgba[border, 0] = 0
    rgba[border, 1] = 255
    rgba[border, 2] = 255
    rgba[border, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def _reference_rgba(reference_2d: np.ndarray, root_mask: np.ndarray) -> Image.Image:
    """Convert a u16 grayscale reference slice -> RGBA PNG with root-mask alpha.

    Per-slice min-max stretch matches what `commands.rs:normalize_volume_to_u8`
    does so the rendered intensities track the desktop build."""
    arr = reference_2d.astype(np.float32)
    max_val = float(arr.max())
    if max_val > 0:
        arr = (arr / max_val * 255.0).clip(0, 255)
    gray = arr.astype(np.uint8)
    rgba = np.dstack([gray, gray, gray, root_mask])
    return Image.fromarray(rgba, mode="RGBA")


def _save_png(img: Image.Image, path: Path) -> int:
    """Save img as PNG; return bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    path.write_bytes(data)
    return len(data)


def main() -> int:
    print(f"Loading atlas {ATLAS_NAME}...")
    atlas = BrainGlobeAtlas(ATLAS_NAME)
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, PLANE)
    n_sections = atlas.reference.shape[axis]
    print(f"  shape={atlas.reference.shape} orient={ctx.orientation} axis={axis} n={n_sections}")

    print("Wiping previous bundle (preserves dir, removes prior runs)...")
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    print("Building palette...")
    palette = _coloring_palette(atlas)
    print(f"  {len(palette)} colored structures (ventricles suppressed)")

    print("Writing structures.json...")
    structures_out: dict[str, dict] = {}
    for sid, struct in atlas.structures.items():
        triplet = struct.get("rgb_triplet")
        structures_out[str(sid)] = {
            "id": int(sid),
            "acronym": str(struct.get("acronym", "")),
            "name": str(struct.get("name", "")),
            "rgb_triplet": (
                [int(triplet[0]), int(triplet[1]), int(triplet[2])]
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3
                else None
            ),
            "structure_id_path": [int(x) for x in struct.get("structure_id_path", [])],
        }
    (OUT_ROOT / "structures.json").write_text(
        json.dumps(structures_out, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  {(OUT_ROOT / 'structures.json').stat().st_size / 1024:.1f} KB")

    sections_dir = OUT_ROOT / "sections"
    colored_dir = OUT_ROOT / "colored"
    borders_dir = OUT_ROOT / "borders"

    sections_meta: list[dict] = []
    bytes_sections = bytes_colored = bytes_borders = 0
    print(f"Rendering {n_sections} sections (this takes a minute)...")
    for idx in range(n_sections):
        ap_mm = float(index_to_position_mm(atlas, idx, plane=PLANE))
        annotation_2d = orient_slice_for_display(
            np.take(atlas.annotation, idx, axis=axis), PLANE
        )
        reference_2d = orient_slice_for_display(
            np.take(atlas.reference, idx, axis=axis), PLANE
        )
        root_mask = _root_mask_for_slice(annotation_2d)

        name = f"{idx:03d}.png"
        bytes_sections += _save_png(
            _reference_rgba(reference_2d, root_mask), sections_dir / name
        )
        bytes_colored += _save_png(
            _colored_slice_rgba(annotation_2d, palette), colored_dir / name
        )
        bytes_borders += _save_png(_border_slice(annotation_2d), borders_dir / name)

        sections_meta.append(
            {
                "idx": idx,
                "mm": round(ap_mm, 4),
                "section": f"sections/{name}",
                "colored": f"colored/{name}",
                "borders": f"borders/{name}",
            }
        )
        if idx % 50 == 0 or idx == n_sections - 1:
            print(f"  [{idx + 1}/{n_sections}] AP={ap_mm:.3f} mm")

    print("Copying root mesh...")
    root_id = 997  # Allen mouse root structure
    src_obj = atlas.root_dir / "meshes" / f"{root_id}.obj"
    if not src_obj.exists():
        raise FileNotFoundError(f"Root mesh missing at {src_obj}")
    dst_mesh = OUT_ROOT / "mesh" / "root.obj"
    dst_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_obj, dst_mesh)
    print(f"  mesh: {dst_mesh.stat().st_size / 1024 / 1024:.1f} MB (uncompressed OBJ)")

    print("Writing manifest.json...")
    manifest = {
        "atlas": ATLAS_NAME,
        "plane": PLANE,
        "orientation": ctx.orientation,
        "shape": list(atlas.reference.shape),
        "resolution_um": list(atlas.resolution),
        "ap_axis_index": ctx.ap_axis_index,
        "ap_resolution_mm": ctx.ap_resolution_mm,
        "ap_min_mm": float(index_to_position_mm(atlas, 0, plane=PLANE)),
        "ap_max_mm": float(index_to_position_mm(atlas, n_sections - 1, plane=PLANE)),
        "n_sections": n_sections,
        "root_structure_id": root_id,
        "mesh_file": "mesh/root.obj",
        "structures_file": "structures.json",
        "sections": sections_meta,
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  {(OUT_ROOT / 'manifest.json').stat().st_size / 1024:.1f} KB")

    total_mb = (
        bytes_sections + bytes_colored + bytes_borders
        + (OUT_ROOT / "structures.json").stat().st_size
        + (OUT_ROOT / "manifest.json").stat().st_size
        + dst_mesh.stat().st_size
    ) / 1024 / 1024
    print()
    print("=" * 60)
    print("DONE.")
    print(f"  sections/ = {bytes_sections / 1024 / 1024:.2f} MB")
    print(f"  colored/  = {bytes_colored / 1024 / 1024:.2f} MB")
    print(f"  borders/  = {bytes_borders / 1024 / 1024:.2f} MB")
    print(f"  mesh.obj  = {dst_mesh.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  TOTAL     = {total_mb:.2f} MB")
    print(f"Output: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
