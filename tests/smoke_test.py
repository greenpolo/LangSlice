"""Comprehensive smoke test for all langslice modules."""

# Version
from langslice import __version__
print(f"Version: {__version__}")

# Atlas module
from langslice.atlas import load_atlas, position_mm_to_index, index_to_position_mm, get_position_range_mm
print("Atlas module OK")

# VLM module
from langslice.vlm.config import get_api_key, get_client, MODEL_NAME, THINKING_LEVEL
print(f"VLM config OK: model={MODEL_NAME} thinking={THINKING_LEVEL}")

from langslice.vlm.estimator import estimate_ap, estimate_affine, APResult, AffineResult, _retry_generate
print("VLM estimator OK (with retry logic)")

# Registration module
from langslice.registration import affine_matrix_from_legacy_params, identity_affine_matrix
print("Registration module OK")

# Export module
from langslice.export import (
    compute_anchoring, build_quint_export, save_quint_json, export_to_dict,
    AnchoringVector, SliceExport, QUINTExport, _resolve_target
)
print("Export module OK")

# Image prep module
from langslice.image_prep import detect_pixel_size_um, load_image_state, prepare_image_for_vlm
print("Image prep OK")

# Test anchoring computation
anch = compute_anchoring(
    position_mm=1.0,
    atlas_shape=(528, 320, 456),
    atlas_resolution=(25.0, 25.0, 25.0),
    image_width=1024,
    image_height=670,
    affine_matrix=identity_affine_matrix(),
)
print(f"Anchoring: o=({anch.ox:.1f}, {anch.oy:.1f}, {anch.oz:.1f}), u=({anch.ux:.1f}, {anch.uy:.1f}, {anch.uz:.1f}), v=({anch.vx:.1f}, {anch.vy:.1f}, {anch.vz:.1f})")

# Test target resolution
t = _resolve_target("allen_mouse_25um")
print(f"Target: {t}")

# Test full export dict
exp = build_quint_export(
    filename="test_slice.png", position_mm=1.0, atlas_name="allen_mouse_25um",
    atlas_shape=(528, 320, 456), atlas_resolution=(25.0, 25.0, 25.0),
    image_width=1024, image_height=670,
    affine_matrix=affine_matrix_from_legacy_params(
        image_width=1024,
        image_height=670,
        rotation_deg=2.5,
        translate_x_pct=1.0,
        translate_y_pct=-0.5,
    ),
)
d = export_to_dict(exp)
assert d["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
assert d["slices"][0]["filename"] == "test_slice.png"
assert len(d["slices"][0]["anchoring"]) == 9
print(f"Export dict OK: target={d['target']}, anchoring has {len(d['slices'][0]['anchoring'])} elements")

result = AffineResult.from_legacy_params(
    image_width=1024,
    image_height=670,
    rotation_deg=2.5,
    translate_x_pct=1.0,
    translate_y_pct=-0.5,
    backend="test",
    reasoning="synthetic",
)
assert result.matrix.shape == (3, 3)
assert result.output_size == (1024, 670)
print(f"AffineResult OK: backend={result.backend}, output={result.output_size}")

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
