"""StackState serialization round-trip."""

from langslice.linear.whole_brain.state import (
    POSITION_SOURCES,
    BrainConfig,
    SliceState,
    StackState,
)


def _state() -> StackState:
    return StackState(
        image_folder="/tmp/brain",
        atlas="allen_mouse_25um",
        plane="coronal",
        axis_directions={"ap": "anterior_to_posterior"},
        interval_mm=0.2,
        thickness_mm=0.05,
        keep_order=False,
        oblique_angles_deg={"dv": 3.5, "ml": -1.0},
        interval_breaks=[4, 9],
        notes=["ingest: 3 slices"],
        contact_sheet="/tmp/brain/contact_sheet.png",
        slices=[
            SliceState(
                id="s1.png",
                index_original=0,
                index_corrected=1,
                flip=True,
                damaged=True,
                damage_note="torn cortex",
                position_mm=1.25,
                position_source="refined",
                affine=[1.0, 0.0, 0.0, 1.0, 3.0, -2.0],
                interactive_transform={"rotation_deg": 2.0, "scale_x": 1.01},
                confidence="medium",
                caveats=["damaged: transform is approximate"],
            ),
            SliceState(id="s2.png", index_original=1, index_corrected=0),
        ],
        completed_nodes=["ingest", "survey"],
        node_cycles={"ingest": 1, "survey": 2},
    )


def test_stack_state_roundtrip():
    original = _state()
    restored = StackState.from_dict(original.to_dict())
    assert restored == original


def test_every_position_source_survives_a_round_trip():
    """Seeding writes 'anchor' and 'interpolated'; both must serialize as-is."""
    assert "interpolated" in POSITION_SOURCES
    state = StackState(
        slices=[
            SliceState(
                id=f"s{index}.png",
                index_original=index,
                index_corrected=index,
                position_mm=float(index),
                position_source=source,
            )
            for index, source in enumerate(POSITION_SOURCES)
        ]
    )
    restored = StackState.from_dict(state.to_dict())
    assert [s.position_source for s in restored.slices] == list(POSITION_SOURCES)


def test_from_dict_ignores_unknown_keys():
    payload = _state().to_dict()
    payload["future_field"] = "whatever"
    payload["slices"][0]["future_slice_field"] = 1
    restored = StackState.from_dict(payload)
    assert restored.slices[0].id == "s1.png"
    assert restored.atlas == "allen_mouse_25um"


def test_from_dict_defaults_missing_fields():
    restored = StackState.from_dict({"atlas": "allen_mouse_25um"})
    assert restored.slices == []
    assert restored.completed_nodes == []
    assert restored.plane == "coronal"


def test_in_order_uses_corrected_index():
    state = _state()
    assert [s.id for s in state.in_order()] == ["s2.png", "s1.png"]


def test_mark_complete_is_idempotent():
    state = StackState()
    state.mark_complete("ingest")
    state.mark_complete("ingest")
    assert state.completed_nodes == ["ingest"]


def test_config_unit_conversion():
    config = BrainConfig(image_folder="/tmp", thickness_um=50, interval_um=200)
    assert config.thickness_mm == 0.05
    assert config.interval_mm == 0.2
