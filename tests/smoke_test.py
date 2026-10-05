"""Smoke checks that public imports and core plumbing still work."""

from __future__ import annotations

import importlib

import numpy as np

from langslice.core.nonlinear import (
    AffineResult,
    affine_matrix_from_legacy_params,
    identity_affine_matrix,
)


def test_public_module_imports() -> None:
    importlib.import_module("langslice")
    importlib.import_module("langslice.core.atlas")
    importlib.import_module("langslice.providers.vlm_config")
    importlib.import_module("langslice.linear")
    importlib.import_module("langslice.core.nonlinear")
    importlib.import_module("langslice.job.quint")
    importlib.import_module("langslice.core.image_prep")
    importlib.import_module("langslice.cli")


def test_affine_result_constructor_smoke() -> None:
    result = AffineResult(
        matrix=affine_matrix_from_legacy_params(
            image_width=1024,
            image_height=670,
            rotation_deg=2.5,
            translate_x_pct=1.0,
            translate_y_pct=-0.5,
        ),
        source_size=(1024, 670),
        output_size=(1024, 670),
        backend="test",
        reasoning="synthetic",
    )
    assert result.matrix.shape == (3, 3)
    assert result.output_size == (1024, 670)
    assert np.allclose(identity_affine_matrix(), np.eye(3, dtype=np.float64))
