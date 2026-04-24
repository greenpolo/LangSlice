"""Smoke checks that public imports and core plumbing still work."""

from __future__ import annotations

import importlib

import numpy as np

from langslice_harness.export import build_quint_export, export_to_dict
from langslice_harness.registration import (
    AffineResult,
    affine_matrix_from_legacy_params,
    identity_affine_matrix,
)


def test_public_module_imports() -> None:
    importlib.import_module("langslice_harness")
    importlib.import_module("langslice_harness.atlas")
    importlib.import_module("langslice_harness.vlm_config")
    importlib.import_module("langslice_harness.estimation")
    importlib.import_module("langslice_harness.registration")
    importlib.import_module("langslice_harness.export")
    importlib.import_module("langslice_harness.image_prep")
    importlib.import_module("langslice_harness.cli")


def test_quint_export_smoke_payload() -> None:
    exp = build_quint_export(
        filename="test_slice.png",
        position_mm=1.0,
        atlas_name="allen_mouse_25um",
        atlas_shape=(528, 320, 456),
        atlas_resolution=(25.0, 25.0, 25.0),
        image_width=1024,
        image_height=670,
        affine_matrix=affine_matrix_from_legacy_params(
            image_width=1024,
            image_height=670,
            rotation_deg=2.5,
            translate_x_pct=1.0,
            translate_y_pct=-0.5,
        ),
    )
    payload = export_to_dict(exp)
    assert payload["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
    assert payload["slices"][0]["filename"] == "test_slice.png"
    assert len(payload["slices"][0]["anchoring"]) == 9


def test_affine_result_constructor_smoke() -> None:
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
    assert np.allclose(identity_affine_matrix(), np.eye(3, dtype=np.float64))
