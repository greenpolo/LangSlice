"""Script-style checks for the new registration runtime."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image

import langslice.registration.runtime as runtime
from langslice.registration.types import RegistrationCorrespondence


def fake_estimate_registration_correspondences(image, *, atlas_name, position_mm, on_progress=None):
    _ = image, atlas_name, position_mm
    if on_progress is not None:
        on_progress("fake correspondences")
    return [
        RegistrationCorrespondence(slice_xy=(20.0, 20.0), atlas_xy=(10.0, 10.0), label="a", confidence="high"),
        RegistrationCorrespondence(slice_xy=(60.0, 22.0), atlas_xy=(50.0, 12.0), label="b", confidence="high"),
        RegistrationCorrespondence(slice_xy=(98.0, 25.0), atlas_xy=(88.0, 15.0), label="c", confidence="high"),
        RegistrationCorrespondence(slice_xy=(24.0, 70.0), atlas_xy=(14.0, 60.0), label="d", confidence="high"),
        RegistrationCorrespondence(slice_xy=(62.0, 72.0), atlas_xy=(52.0, 62.0), label="e", confidence="high"),
        RegistrationCorrespondence(slice_xy=(95.0, 78.0), atlas_xy=(85.0, 68.0), label="f", confidence="high"),
    ]


original_agent = runtime.estimate_registration_correspondences

try:
    runtime.estimate_registration_correspondences = fake_estimate_registration_correspondences
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["LANGSLICE_VLM_DEBUG_DIR"] = tmpdir
        result = runtime.estimate_registration(
            Image.new("RGB", (120, 100), (255, 255, 255)),
            atlas_name="allen_mouse_25um",
            position_mm=1.0,
        )
        assert result.affine_result.backend == "registration_agent_affine"
        assert result.nonlinear_result.backend == "tps"
        assert result.qc_state in {"accepted", "review"}
        assert result.debug_dir is not None
        run_dir = Path(result.debug_dir)
        assert (run_dir / "registration.json").exists()
        assert (run_dir / "slice_landmarks.png").exists()
        del os.environ["LANGSLICE_VLM_DEBUG_DIR"]
finally:
    runtime.estimate_registration_correspondences = original_agent

print("Registration runtime OK")
