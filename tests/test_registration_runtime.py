"""Checks for the registration runtime."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import langslice.registration.runtime as runtime
from langslice.registration.types import RegistrationCorrespondence


def fake_estimate_registration_correspondences(
    image,
    *,
    atlas_name,
    position_mm,
    target_landmark_count,
    workflow="single_pass",
    show_atlas_borders=True,
    on_progress=None,
    on_trace=None,
    debug_dir=None,
):
    _ = image, atlas_name, position_mm
    _ = on_trace, debug_dir
    assert target_landmark_count == 14
    assert workflow == "single_pass"
    assert show_atlas_borders is True
    if on_progress is not None:
        on_progress("fake correspondences")
    return [
        RegistrationCorrespondence(
            slice_xy=(4.0, 4.0),
            atlas_xy=(8.0, 8.0),
            label="e1",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(116.0, 4.0),
            atlas_xy=(448.0, 8.0),
            label="e2",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(4.0, 96.0),
            atlas_xy=(8.0, 312.0),
            label="e3",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(116.0, 96.0),
            atlas_xy=(448.0, 312.0),
            label="e4",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(60.0, 4.0),
            atlas_xy=(230.0, 8.0),
            label="e5",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(60.0, 96.0),
            atlas_xy=(230.0, 312.0),
            label="e6",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(4.0, 50.0),
            atlas_xy=(8.0, 160.0),
            label="e7",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(116.0, 50.0),
            atlas_xy=(448.0, 160.0),
            label="e8",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(20.0, 20.0),
            atlas_xy=(80.0, 70.0),
            label="i1",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(60.0, 22.0),
            atlas_xy=(220.0, 75.0),
            label="i2",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(98.0, 25.0),
            atlas_xy=(360.0, 85.0),
            label="i3",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(24.0, 70.0),
            atlas_xy=(95.0, 240.0),
            label="i4",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(62.0, 72.0),
            atlas_xy=(230.0, 250.0),
            label="i5",
            confidence="high",
        ),
        RegistrationCorrespondence(
            slice_xy=(95.0, 78.0),
            atlas_xy=(350.0, 260.0),
            label="i6",
            confidence="high",
        ),
    ]


def test_registration_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runtime,
        "estimate_registration_correspondences",
        fake_estimate_registration_correspondences,
    )
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", str(tmp_path))
    result = runtime.estimate_registration(
        Image.new("RGB", (120, 100), (255, 255, 255)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        target_landmark_count=14,
    )
    assert result.affine_result.backend == "landmark_affine"
    assert result.affine_result.provenance.get("transform_direction") == "atlas_to_slice"
    assert result.nonlinear_result.backend == "tps"
    assert result.qc_state in {"accepted", "review"}
    assert result.debug_dir is not None
    assert result.annotation_session is not None
    assert result.annotation_session.workflow == "single_pass"
    assert result.annotation_session.target_count == 14
    assert len(result.annotation_session.atlas_annotations) == 14
    assert len(result.annotation_session.slice_annotations) == 14
    assert result.annotation_session.slice_annotations[0].normalized_yx is None
    run_dir = Path(result.debug_dir)
    assert (run_dir / "registration.json").exists()
    assert (run_dir / "slice_landmarks.png").exists()


def test_registration_runtime_uses_existing_debug_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runtime,
        "estimate_registration_correspondences",
        fake_estimate_registration_correspondences,
    )
    existing = tmp_path / "shared_run"
    existing.mkdir()
    result = runtime.estimate_registration(
        Image.new("RGB", (120, 100), (255, 255, 255)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        target_landmark_count=14,
        debug_dir=str(existing),
    )
    assert result.debug_dir == str(existing / "registration")
    assert (existing / "registration" / "registration.json").exists()
