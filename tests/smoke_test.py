"""Comprehensive smoke test for all langslice modules."""

# Version
from langslice import __version__
print(f"Version: {__version__}")

# Atlas module
from langslice.atlas import load_atlas, get_origin_index, ap_mm_to_index, get_ap_range
print("Atlas module OK")

# VLM module
from langslice.vlm.config import get_api_key, get_client, MODEL_NAME, THINKING_BUDGET
print(f"VLM config OK: model={MODEL_NAME} budget={THINKING_BUDGET}")

from langslice.vlm.estimator import estimate_ap, estimate_affine, APResult, AffineResult, _retry_generate
print("VLM estimator OK (with retry logic)")

# Export module
from langslice.export import (
    compute_anchoring, build_quint_export, save_quint_json, export_to_dict,
    AnchoringVector, SliceExport, QUINTExport, _resolve_target
)
print("Export module OK")

# Test anchoring computation
anch = compute_anchoring(
    ap_mm=1.0,
    atlas_shape=(528, 320, 456),
    atlas_resolution=(25.0, 25.0, 25.0),
    origin_index=264,
    image_width=1024,
    image_height=670,
)
print(f"Anchoring: o=({anch.ox:.1f}, {anch.oy:.1f}, {anch.oz:.1f}), u=({anch.ux:.1f}, {anch.uy:.1f}, {anch.uz:.1f}), v=({anch.vx:.1f}, {anch.vy:.1f}, {anch.vz:.1f})")

# Test target resolution
t = _resolve_target("allen_mouse_25um")
print(f"Target: {t}")

# Test full export dict
exp = build_quint_export(
    filename="test_slice.png", ap_mm=1.0, atlas_name="allen_mouse_25um",
    atlas_shape=(528, 320, 456), atlas_resolution=(25.0, 25.0, 25.0),
    origin_index=264, image_width=1024, image_height=670,
    rotation_deg=2.5, translate_x_pct=1.0, translate_y_pct=-0.5,
)
d = export_to_dict(exp)
assert d["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
assert d["slices"][0]["filename"] == "test_slice.png"
assert len(d["slices"][0]["anchoring"]) == 9
print(f"Export dict OK: target={d['target']}, anchoring has {len(d['slices'][0]['anchoring'])} elements")

# GUI modules
from langslice.gui.theme import STYLESHEET, ACCENT
print(f"Theme OK: accent={ACCENT}")

from langslice.gui.atlas_viewer import AtlasViewer
print("AtlasViewer OK")

from langslice.gui.main_window import MainWindow, run
print("MainWindow OK")

# CLI
from langslice.cli import main
print("CLI OK")

print()
print("=== ALL MODULES VERIFIED ===")
