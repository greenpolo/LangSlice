from __future__ import annotations

import json
from typing import Any

import pytest

from langslice.core.stack.affine import Calibration, FitRegions, PhysicalAffine, StoredAffine
from langslice.core.stack.deformation import DeformationRef, LandmarkWarp
from langslice.core.stack.images import ImageRef, LayerRef, MaskRef
from langslice.core.stack.landmarks import Endpoint, Landmarks, Pair
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.core.stack.trace import BorderTrace
from langslice.job.state_file import stack_from_json, stack_to_json

END = {"picture": 1, "registration_context": {"plane": "coronal"}, "geometry": {"size": [4, 4]}}
LAYER = {"name": "n", "kind": "points", "frame": "f", "file": "f", "tool": "t", "call": "c"}
MASK = {"kind": "em", "section_image": "s", "atlas_image": "a", "params": {}, "inputs": "i"}


def endpoint() -> Endpoint:
    return Endpoint(canonical_xy=(0.5, 2), pixel_xy=(10, 20.5), side_note="e", **END)


def full_section() -> Section:
    pair = Pair(
        id="L1",
        atlas=endpoint(),
        atlas_native_xy=(1, 2),
        section=endpoint(),
        section_fraction_xy=(0.25, 0.75),
        plane={"angles": [0, 0]},
        stale=False,
        pair_note=None,
    )
    warp = LandmarkWarp(
        revision="r", pairs={"L1": "k"}, folds={}, baseline_key=None, baseline_steps=()
    )
    return Section(
        id="a.tif",
        index_original=0,
        index_corrected=0,
        flip=True,
        rotation_deg=90,
        damaged_regions=["CTX:left"],
        position_mm=5,
        position_source="input",
        transform=StoredAffine(
            kind="elastix",
            params=(1, 0, 0.5, 0, 1.0, 0),
            physical=PhysicalAffine(rotation_deg=0, scale_x=1.0, pivot=(0.5, 0.5), knob_note=1),
            calibration=Calibration(section_um_per_px=None, source="supplied"),
            mirrored=False,
            regions=FitRegions(include=("CTX",), exclude=()),
            transform_note="t",
        ),
        border_trace=BorderTrace(
            id="a.tif", geometry_fingerprint="g", status="ok", raw_size=(8, 8), trace_note=3
        ),
        deformation=DeformationRef(
            key="k",
            linear_key="l",
            steps=({"method": "fireants_syn"},),
            landmark=warp,
            summary={"folds": 0},
            deformation_note=[1],
        ),
        landmarks=Landmarks(next_id=2, pairs=(pair,)),
        layers=[LayerRef(id="t/1", created="x", **LAYER)],
        masks={"m": MaskRef(file="f", left_out=0.1, mask_note=0, **MASK)},
        images={"i": ImageRef(source={"opening": "plain"}, file="f", digest="d", made="m")},
        angles=(1.5, 2),
    )


def test_every_part_survives_a_round_trip_with_its_unknown_keys() -> None:
    plain = Section(id="b.tif", index_original=1, index_corrected=1, position_mm=6.5)
    stack = Stack(atlas="x", plane="coronal", slices=[full_section(), plain])
    written = json.loads(json.dumps(stack_to_json(stack)))
    assert stack_from_json(written) == stack
    row = written["slices"][0]
    assert list(row["transform"])[-1] == "transform_note"
    assert list(row["deformation"])[-1] == "deformation_note"
    assert list(row["transform"]["physical"])[-1] == "knob_note"
    assert list(row["landmarks"]["pairs"][0])[-1] == "pair_note"
    assert list(row["landmarks"]["pairs"][0]["atlas"])[-1] == "side_note"
    assert isinstance(row["transform"]["params"][0], int)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("position_mm",), "5"),
        (("flip",), 1),
        (("transform", "params"), [1, True]),
        (("transform", "physical", "pivot"), [0.5]),
        (("deformation", "key"), 3),
    ],
)
def test_a_wrong_typed_value_names_its_key(path: tuple[str, ...], value: Any) -> None:
    written = json.loads(json.dumps(stack_to_json(Stack(slices=[full_section()]))))
    holder = written["slices"][0]
    for step in path[:-1]:
        holder = holder[step]
    holder[path[-1]] = value
    with pytest.raises(ValueError, match=path[-1]):
        stack_from_json(written)
