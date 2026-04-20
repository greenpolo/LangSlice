from langslice.harness.estimation.session import (
    ARTIFACT_TARGET,
    build_initial_state,
)


def test_initial_state_single_slice():
    state = build_initial_state(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        pos_lo=0.0, pos_hi=13.2,
        n_slices=1, interval_mm=0.0, thickness_um=50,
        max_iterations=20,
    )
    assert state["atlas"] == "allen_mouse_25um"
    assert state["plane"] == "coronal"
    assert state["axis_label"] == "AP"
    assert state["n_slices"] == 1
    assert state["fetched_positions"] == []
    assert state["saw_broad_sweep"] is False
    assert state["saw_narrow_sweep"] is False
    assert state["submit_attempts"] == 0
    assert state["result"] is None


def test_artifact_target_constant():
    assert ARTIFACT_TARGET == "target"
