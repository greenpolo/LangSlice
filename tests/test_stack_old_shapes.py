"""Section rows in the shapes earlier LangSlice versions wrote read and write back.

The fixtures are small synthetic rows that copy the old forms found in job
folders: an imported affine with its ``in_plane`` knobs, an Elastix affine
with its ``iou``, a section image with ``weights_used``, a landmark warp
saved with automatic holds and a null ``landmark_baseline_key``, and the
``keep_linear`` marker a format-3 job wrote for a section left linear.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from langslice.core.fits.deformation import linear_key
from langslice.core.stack.rows import section_from_json, section_to_json
from langslice.job.state_file import stack_from_json, stack_to_json

ANGLES = {"pitch": -1.5, "yaw": -0.9}

IMPORTED = {
    "kind": "imported",
    "params": [1.026, 0.0397, -0.0788, -0.0440, 1.0088, -0.0451],
    "physical": {"rotation_deg": 2.266, "scale_x": 1.027, "scale_y": 1.0097,
                 "translate_x_mm": -0.2115, "translate_y_mm": -0.2665,
                 "shear": 0.003, "pivot": [0.5, 0.5]},
    "in_plane": {"rotation_deg": 2.266, "scale_x": 1.027, "scale_y": 1.0097,
                 "shear": 0.003},
    "mirrored": False,
    "calibration": {"section_um_per_px": 8.99764, "source": "host"},
    "note": "imported from s001.png",
}

ELASTIX = {
    "kind": "elastix",
    "params": [1.0012, 0.0247, -0.0073, -0.0400, 1.0804, -0.0259],
    "physical": {"rotation_deg": 1.715, "scale_x": 1.0017, "scale_y": 1.0809,
                 "translate_x_mm": 0.0411, "translate_y_mm": -0.0304,
                 "shear": 0.0006, "pivot": [0.5, 0.5]},
    "iou": 0.964,
    "calibration": {"section_um_per_px": 14.1393, "source": "estimated"},
    "mirrored": False,
}

MADE_IMAGE = {
    "source": {"recipe": {"weights": "auto", "clahe_clip": 12.0, "clahe_tiles": 8,
                          "n4": True, "denoise": True, "contrast": None,
                          "gamma": None, "invert": False}},
    "file": "sections/s001/images/hc.png",
    "digest": "576d0da1",
    "made": "2026-10-27T19:27:51.686128+00:00",
    "weights_used": {"red": 0.3334, "green": 0.3333, "blue": 0.3333},
}


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _placement_key(state: dict[str, Any], row: dict[str, Any]) -> str:
    """The linear key as main computed it from a row's plain JSON."""
    transform = row.get("transform") or {}
    angles = row["cutting_angles_deg"]
    return _digest({
        "atlas": state["atlas"], "plane": state["plane"],
        "position_mm": row.get("position_mm"),
        "angles": [float(angles["pitch"]), float(angles["yaw"])],
        "flip": row.get("flip", False), "rotation_deg": row.get("rotation_deg", 0),
        "transform": {key: transform.get(key) for key in ("params", "calibration")},
    })


def _row(section_id: str, index: int, transform: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": section_id, "index_original": index, "index_corrected": index,
        "flip": False, "rotation_deg": 0, "damaged_regions": [], "damage_note": "",
        "position_mm": 2.6 + index * 0.1, "transform": transform,
        "border_trace": None, "deformation": None, "landmarks": {},
        "layers": [], "masks": {}, "images": {}, "caveats": [],
        "cutting_angles_deg": dict(ANGLES),
    }


def _state() -> dict[str, Any]:
    """A two-section state: a landmark warp with holds on an imported affine,
    and a kept-linear marker on an Elastix affine."""
    state: dict[str, Any] = {
        "image_folder": "/data/brain", "atlas": "allen_mouse_25um", "plane": "coronal",
        "slices": [_row("s001.png", 0, IMPORTED), _row("s002.png", 1, ELASTIX)],
    }
    warped, kept = state["slices"]
    warped["images"] = {"hc": MADE_IMAGE}
    warped["deformation"] = {
        "key": "98d52480" * 8,
        "record": "sections/s001/deformable/98d52480673475c4f1fbb397",
        "linear_key": _placement_key(state, warped),
        "summary": {"displacement_mm": {"max": 0.576, "median": 0.0},
                    "fold_fraction": 0.0, "flags": []},
        "inverse_source": "exact inverse on the same landmark triangles",
        "landmark_revision": "7fe33bf9" * 8,
        "landmark_pairs": {"L1": "8c553607" * 8, "L2": "0d7f3c56" * 8},
        "landmark_folds": {},
        "landmark_holds": [[0.2143, 0.3], [0.5, 0.3]],
        "landmark_baseline_key": None,
        "landmark_baseline_steps": [],
        "steps": [{"method": "landmark_triangulation", "start": "linear",
                   "landmark_revision": "7fe33bf9" * 8, "landmark_count": 2}],
    }
    kept["deformation"] = {
        "keep_linear": "The linear placement already matches.",
        "linear_key": _placement_key(state, kept),
    }
    return state


@pytest.mark.parametrize("index", [0, 1])
def test_an_old_row_reads_and_writes_back_with_its_extra_keys(index: int):
    row = _state()["slices"][index]
    assert section_to_json(section_from_json(copy.deepcopy(row))) == row


def test_the_old_keys_stay_where_they_were_written():
    state = _state()
    written = stack_to_json(stack_from_json(copy.deepcopy(state)))
    assert json.dumps(written["slices"][0]["deformation"]) == json.dumps(
        state["slices"][0]["deformation"])
    assert json.dumps(written["slices"][1]["deformation"]) == json.dumps(
        state["slices"][1]["deformation"])


def test_the_stored_linear_key_is_the_one_the_placement_gives():
    state = _state()
    stack = stack_from_json(copy.deepcopy(state))
    for section in stack.slices:
        assert section.deformation is not None
        assert linear_key(stack, section) == section.deformation.linear_key


def test_a_kept_linear_marker_holds_no_saved_fit():
    stack = stack_from_json(_state())
    held = stack.slices[1].deformation
    assert held is not None and held.key is None and held.steps is None
    assert held.kept_linear == "The linear placement already matches."
    assert stack.slices[1].deformation_key is None


def test_a_part_that_does_not_read_names_its_section():
    row = _state()["slices"][1]
    del row["deformation"]["keep_linear"]
    with pytest.raises(ValueError, match="section s002.png, deformation: key"):
        section_from_json(row)
