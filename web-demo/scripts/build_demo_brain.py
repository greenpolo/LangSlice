"""Pre-process Walsh Lab brain M01 for the LangSlice GH Pages demo.

Inputs (read-only):
  C:/WalshLab/SIArevision/mouseimages/M01/M01_NN.tif   (29 RGB whole-slide TIFs)
  C:/WalshLab/SIArevision/EXACTbregmapoints.abba       (ABBA registration state, ZIP)
  C:/WalshLab/SIArevision/EXACTbregmapointsPart2.abba  (second part, also ZIP)

Outputs:
  web-demo/public/demo_brain/slice_NN.png   (29 RGB PNGs, long-edge <= 2048 px,
                                              brightness/exposure boosted)
  web-demo/public/demo_brain/manifest.json  (brain_id, atlas, plane,
                                              per-section ground_truth_mm)

The .abba files are JSON-in-ZIP: ``state.json`` holds a ``slices_state_list``
where each slice has CreateSliceAction (source_indexes -> sources.json) and
zero-or-more MoveSliceAction (final AP location in mm). We extract the last
MoveSliceAction.location as the per-slice ground-truth AP. ABBA's location
convention here is atlas-mm-from-anterior-pole (matches what the harness's
``index_to_position_mm`` produces from BrainGlobe).

Run:
    python web-demo/scripts/build_demo_brain.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageEnhance

BRAIN_ID = "M01"
ATLAS = "allen_mouse_25um"
PLANE = "coronal"
TIF_DIR = Path("C:/WalshLab/SIArevision/mouseimages") / BRAIN_ID
ABBA_FILES = [
    Path("C:/WalshLab/SIArevision/EXACTbregmapoints.abba"),
    Path("C:/WalshLab/SIArevision/EXACTbregmapointsPart2.abba"),
]
OUT_ROOT = Path(__file__).resolve().parents[1] / "public" / "demo_brain"

LONG_EDGE = 2048           # downsample target
BRIGHTNESS = 1.30          # PIL ImageEnhance.Brightness factor
CONTRAST = 1.20            # PIL ImageEnhance.Contrast factor
PERCENTILE_STRETCH = (1.0, 99.5)  # clip + rescale before brightness/contrast

# Port of `langslice_harness.image_prep.adaptive_preprocess` — per-channel
# CLAHE, weighted grayscale blend that leans on the (DAPI / structural) blue
# channel, then an adaptive brightness pull to a target mean. Defaults must
# match the Python CLI verbatim so demo previews look like what the model
# sees at inference time.
CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_GRID = (8, 8)
CLAHE_CHANNEL_WEIGHTS = (0.15, 0.15, 0.70)
CLAHE_TARGET_BRIGHTNESS = 90.0
CLAHE_MAX_BOOST = 3.0
CLAHE_BRAIN_MASK_THRESHOLD = 10

# Section IDs to drop from the bundle. The first five M01 sections are
# discarded slabs we don't want to ship.
DROP_SECTION_IDS: set[int] = {1, 2, 3, 4, 5}


_SOURCE_NAME_RE = re.compile(rf"^{BRAIN_ID}_(\d+)\.tif", re.IGNORECASE)


def _parse_abba(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (sources_list, slices_state_list) from one .abba ZIP."""
    with zipfile.ZipFile(path) as z:
        with z.open("sources.json") as f:
            sources = json.load(f)
        with z.open("state.json") as f:
            state = json.load(f)
    return sources, state.get("slices_state_list", [])


def _index_to_section_id(sources: list[dict]) -> dict[int, int]:
    """Map global source_index -> M01 section number (parsed from source_name).

    sources entries look like ``{"source_id": 738, "source_name": "M01_05.tif-Red", ...}``.
    Three sources per slice (RGB), all sharing the same section number."""
    out: dict[int, int] = {}
    for src in sources:
        name = str(src.get("source_name", ""))
        m = _SOURCE_NAME_RE.match(name)
        if not m:
            continue
        section_id = int(m.group(1))
        out[int(src["source_id"])] = section_id
    return out


def _last_move_location(actions: list[dict]) -> float | None:
    """Return the location of the last MoveSliceAction in this slice's actions."""
    last: float | None = None
    for action in actions:
        if action.get("type") == "MoveSliceAction":
            loc = action.get("location")
            if isinstance(loc, (int, float)):
                last = float(loc)
    if last is None:
        for action in actions:
            if action.get("type") == "CreateSliceAction":
                loc = action.get("original_location")
                if isinstance(loc, (int, float)):
                    last = float(loc)
                    break
    return last


def _build_section_to_mm() -> dict[int, float]:
    """Walk both .abba files and build {M01 section number -> AP mm}."""
    section_to_mm: dict[int, float] = {}
    for abba_path in ABBA_FILES:
        if not abba_path.exists():
            print(f"[warn] ABBA file missing: {abba_path}")
            continue
        sources, slices = _parse_abba(abba_path)
        idx_to_section = _index_to_section_id(sources)
        if not idx_to_section:
            print(f"[info] no {BRAIN_ID} sources in {abba_path.name}")
            continue
        print(f"[info] {abba_path.name}: {len(idx_to_section)} {BRAIN_ID} sources, "
              f"{len(slices)} slices")
        for slice_state in slices:
            actions = slice_state.get("actions", [])
            create = next(
                (a for a in actions if a.get("type") == "CreateSliceAction"), None
            )
            if create is None:
                continue
            indexes = create.get("original_sources", {}).get("source_indexes", [])
            sections_in_slice = {
                idx_to_section[i] for i in indexes if i in idx_to_section
            }
            if not sections_in_slice:
                continue
            mm = _last_move_location(actions)
            if mm is None:
                continue
            for section_id in sections_in_slice:
                if section_id in section_to_mm:
                    continue
                section_to_mm[section_id] = mm
    return section_to_mm


def _percentile_stretch(rgb: np.ndarray) -> np.ndarray:
    """Per-channel 1st/99.5th percentile clip + rescale to 0..255 uint8.

    Lifts dim histology into a usable display range without clipping the
    bright background or losing low-end tissue detail."""
    out = np.empty_like(rgb)
    lo_p, hi_p = PERCENTILE_STRETCH
    for c in range(rgb.shape[-1]):
        ch = rgb[..., c].astype(np.float32)
        lo = float(np.percentile(ch, lo_p))
        hi = float(np.percentile(ch, hi_p))
        if hi <= lo:
            out[..., c] = ch.astype(np.uint8)
            continue
        scaled = (ch - lo) * (255.0 / (hi - lo))
        out[..., c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def _apply_adaptive_preprocess(img: Image.Image) -> Image.Image:
    """Port of `langslice_harness.image_prep.adaptive_preprocess`.

    Per-channel CLAHE, then a blue-weighted grayscale blend (fluorescent
    histology has DAPI in the blue channel — viral tracers in red/green
    are noisier and shouldn't dominate the structural signal), then an
    adaptive brightness pull on the brain mask up to a clip. Output is
    a 3-channel RGB but every channel is the same grayscale value, so
    JPEG/PNG codecs and the model see a true monochrome image."""
    rgb = np.asarray(img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"adaptive_preprocess expects RGB; got shape={rgb.shape}")
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
    )
    r_enh = clahe.apply(r).astype(np.float32)
    g_enh = clahe.apply(g).astype(np.float32)
    b_enh = clahe.apply(b).astype(np.float32)

    wr, wg, wb = CLAHE_CHANNEL_WEIGHTS
    blended = wr * r_enh + wg * g_enh + wb * b_enh
    blended = np.clip(blended, 0, 255)

    brain_mask = blended > CLAHE_BRAIN_MASK_THRESHOLD
    if brain_mask.sum() > 0:
        current_mean = float(blended[brain_mask].mean())
        boost = min(CLAHE_TARGET_BRIGHTNESS / max(current_mean, 1.0), CLAHE_MAX_BOOST)
    else:
        boost = 1.0
    if boost > 1.05:
        blended = blended * boost

    gray = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")


def _process_tif(
    tif_path: Path, out_png: Path, clahe_png: Path
) -> tuple[int, int, int]:
    """Read a TIF, downsample, brightness/contrast boost, save base + CLAHE.

    Returns (orig_long_edge_px, base_bytes, clahe_bytes)."""
    arr = tifffile.imread(str(tif_path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr_max = float(arr.max()) or 1.0
        arr = (arr.astype(np.float32) / arr_max * 255.0).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(
            f"Unexpected channel count for {tif_path.name}: shape={arr.shape}"
        )
    orig_long = max(arr.shape[0], arr.shape[1])

    # Downsample once on the normalized RGB so both base + CLAHE see the
    # same pixel grid (so the CLAHE tile-grid lands in the same anatomy).
    base_norm = Image.fromarray(arr, mode="RGB")
    base_norm.thumbnail((LONG_EDGE, LONG_EDGE), Image.Resampling.LANCZOS)

    # Base display PNG: percentile stretch + brightness/contrast — purely
    # for the lay-viewer preview.
    stretched = _percentile_stretch(np.asarray(base_norm, dtype=np.uint8))
    base_img = Image.fromarray(stretched, mode="RGB")
    base_img = ImageEnhance.Brightness(base_img).enhance(BRIGHTNESS)
    base_img = ImageEnhance.Contrast(base_img).enhance(CONTRAST)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    base_img.save(out_png, format="PNG", optimize=True)

    # CLAHE PNG: faithful port of the harness adaptive_preprocess, applied
    # to the raw (unstretched) downsample so the per-channel CLAHE sees
    # the real signal distribution — not the display-boosted preview.
    clahe_img = _apply_adaptive_preprocess(base_norm)
    clahe_img.save(clahe_png, format="PNG", optimize=True)

    return orig_long, out_png.stat().st_size, clahe_png.stat().st_size


def main() -> int:
    if not TIF_DIR.exists():
        print(f"ERROR: TIF directory not found: {TIF_DIR}")
        return 2

    print(f"Indexing ABBA registrations for brain {BRAIN_ID}...")
    section_to_mm = _build_section_to_mm()
    print(f"  resolved AP for {len(section_to_mm)} {BRAIN_ID} sections")

    tif_paths = sorted(TIF_DIR.glob(f"{BRAIN_ID}_*.tif"))
    print(f"Found {len(tif_paths)} TIFs in {TIF_DIR}")
    if not tif_paths:
        print("ERROR: no TIFs to process")
        return 2

    if OUT_ROOT.exists():
        for f in OUT_ROOT.glob("*.png"):
            f.unlink()
        for f in OUT_ROOT.glob("*.json"):
            f.unlink()
    else:
        OUT_ROOT.mkdir(parents=True)

    sections_meta: list[dict] = []
    skipped: list[str] = []
    total_bytes = 0

    for tif_path in tif_paths:
        m = re.match(rf"{BRAIN_ID}_(\d+)\.tif$", tif_path.name, re.IGNORECASE)
        if not m:
            skipped.append(f"{tif_path.name}: bad filename")
            continue
        section_id = int(m.group(1))
        if section_id in DROP_SECTION_IDS:
            skipped.append(f"{tif_path.name}: section {section_id} is in DROP_SECTION_IDS")
            continue
        ap_mm = section_to_mm.get(section_id)
        if ap_mm is None:
            skipped.append(f"{tif_path.name}: no ABBA AP for section {section_id}")
            continue

        out_png = OUT_ROOT / f"slice_{section_id:02d}.png"
        clahe_png = OUT_ROOT / f"slice_{section_id:02d}_clahe.png"
        try:
            orig_long, out_bytes, clahe_bytes = _process_tif(
                tif_path, out_png, clahe_png
            )
        except Exception as exc:
            skipped.append(f"{tif_path.name}: {type(exc).__name__}: {exc}")
            continue
        total_bytes += out_bytes + clahe_bytes
        sections_meta.append(
            {
                "filename": out_png.name,
                "clahe_filename": clahe_png.name,
                "ground_truth_mm": round(ap_mm, 4),
                "original_index": section_id,
            }
        )
        print(f"  {tif_path.name} ({orig_long}px) -> {out_png.name} "
              f"AP={ap_mm:.3f} mm "
              f"(base {out_bytes / 1024:.0f} KB + clahe {clahe_bytes / 1024:.0f} KB)")

    sections_meta.sort(key=lambda s: s["ground_truth_mm"])

    manifest = {
        "brain_id": BRAIN_ID,
        "atlas": ATLAS,
        "plane": PLANE,
        "ap_convention": "atlas_mm_from_anterior_pole",
        "preprocessing": {
            "long_edge_px": LONG_EDGE,
            "percentile_stretch": list(PERCENTILE_STRETCH),
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "clahe": {
                "recipe": "langslice_harness.image_prep.adaptive_preprocess",
                "clip_limit": CLAHE_CLIP_LIMIT,
                "tile_grid": list(CLAHE_TILE_GRID),
                "channel_weights_rgb": list(CLAHE_CHANNEL_WEIGHTS),
                "target_brightness": CLAHE_TARGET_BRIGHTNESS,
                "max_boost": CLAHE_MAX_BOOST,
                "brain_mask_threshold": CLAHE_BRAIN_MASK_THRESHOLD,
            },
        },
        "dropped_section_ids": sorted(DROP_SECTION_IDS),
        "sections": sections_meta,
        "skipped": skipped,
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("DONE.")
    print(f"  Wrote {len(sections_meta)} PNGs")
    print(f"  Skipped {len(skipped)}")
    for line in skipped:
        print(f"    - {line}")
    print(f"  Total bundle size: {total_bytes / 1024 / 1024:.2f} MB")
    if sections_meta:
        ap_values = [s["ground_truth_mm"] for s in sections_meta]
        print(f"  AP range covered: {min(ap_values):.3f} to {max(ap_values):.3f} mm")
    print(f"Output: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
