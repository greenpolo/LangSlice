"""The ABBA mirror's diff/write logic, against a fake ABBA facade — no JVM.

The mirror imports JPype/scyjava lazily inside the methods that build new
Java objects from scratch (``_register_affine``, ``_delete_last_registration``,
``_mark_batch_boundary``); everything else calls methods on objects the fake
facade already provides, which is what these tests exercise directly. The
two Java-object-building seams are monkeypatched where their bookkeeping
(not their exact Java call) is under test.
"""

from __future__ import annotations

import math
import os

import pytest

from langslice.integrations.abba_linear import (
    FLIP_ROTATION_AXIS,
    QUARTER_TURN_AXIS,
    QUARTER_TURN_SIGN,
    AbbaStackMirror,
)
from langslice.linear.state import SliceState, StackState

# --- a fake ABBA facade ----------------------------------------------------


class FakeAffineTransform3D:
    """Stands in for net.imglib2.realtransform.AffineTransform3D: ``copy()``
    and ``rotate()`` are the only two operations the mirror calls on it."""

    def __init__(self, ops: list[tuple] | None = None) -> None:
        self.ops = list(ops or [])

    def copy(self) -> FakeAffineTransform3D:
        return FakeAffineTransform3D(self.ops)

    def rotate(self, axis: int, angle: float) -> None:
        self.ops.append(("rotate", axis, angle))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeAffineTransform3D) and self.ops == other.ops


class FakeSlice:
    def __init__(self, name: str, z: float) -> None:
        self._name = name
        self._z = z
        self._pretransform = FakeAffineTransform3D()
        self.pretransform_history: list[FakeAffineTransform3D] = []

    def getSlicingAxisPosition(self) -> float:
        return self._z

    def getName(self) -> str:
        return self._name

    def isSelected(self) -> bool:
        return True

    def getNumberOfRegistrations(self) -> int:
        return 0

    def getTransformSourceOrigin(self) -> FakeAffineTransform3D:
        return self._pretransform

    def transformSourceOrigin(self, at3d: FakeAffineTransform3D) -> None:
        self._pretransform = at3d
        self.pretransform_history.append(at3d)


class FakeReslicedAtlas:
    def __init__(self) -> None:
        self.rotate_x: float | None = None
        self.rotate_y: float | None = None

    def setRotateX(self, rad: float) -> None:
        self.rotate_x = rad

    def setRotateY(self, rad: float) -> None:
        self.rotate_y = rad

    def getRotateX(self) -> float:
        return self.rotate_x or 0.0

    def getRotateY(self) -> float:
        return self.rotate_y or 0.0


class FakeMP:
    def __init__(self, slices: list[FakeSlice]) -> None:
        self._slices = slices
        self.moves: list[tuple[FakeSlice, float]] = []
        self._resliced = FakeReslicedAtlas()

    def getSlices(self) -> list[FakeSlice]:
        return list(self._slices)

    def moveSlice(self, slice_obj: FakeSlice, z: float) -> None:
        slice_obj._z = z
        self.moves.append((slice_obj, z))

    def getReslicedAtlas(self) -> FakeReslicedAtlas:
        return self._resliced


class FakeAbba:
    """Mirrors the real Abba's own lifecycle: ``mp`` (MultiSlicePositioner)
    exists from construction, and importing files adds slices to it — it is
    never replaced, which is what lets the mirror capture ``self.mp``
    up front, exactly as it does against a real session."""

    def __init__(self) -> None:
        self.mp = FakeMP([])
        self.state_saves: list[str] = []

    def import_from_files(
        self, filepaths: list[str], z_location: float, z_increment: float
    ) -> None:
        # abba_python's own import_from_files blocks and returns a resolved
        # CommandModule, not a Future -- see abba_linear.py's comment.
        self.mp._slices.extend(
            FakeSlice(os.path.basename(path), z_location + i * z_increment)
            for i, path in enumerate(filepaths)
        )

    def wait_for_end_of_tasks(self) -> None:
        pass

    def state_save(self, path: str) -> None:
        self.state_saves.append(path)


_PATHS = ["/stack/a.png", "/stack/b.png", "/stack/c.png"]


def _stack(n: int = 3) -> StackState:
    return StackState(
        interval_mm=0.2,
        slices=[
            SliceState(id=f"{'abc'[i]}.png", index_original=i, index_corrected=i) for i in range(n)
        ],
    )


def _mirror() -> tuple[AbbaStackMirror, FakeAbba]:
    abba = FakeAbba()
    mirror = AbbaStackMirror(abba, _PATHS)
    return mirror, abba


# --- construction ----------------------------------------------------------


def test_construction_maps_slices_by_import_order_and_falls_back_the_offset():
    mirror, abba = _mirror()
    assert set(mirror._slice_by_id) == {"a.png", "b.png", "c.png"}
    assert all(mirror._has_affine[name] is False for name in mirror._slice_by_id)
    # no JPype in this environment: measure_axis_offset must fall back, loudly,
    # rather than raise.
    assert (mirror._a, mirror._b) == (1.0, 1.0)


# --- position moves ----------------------------------------------------------


def test_on_write_moves_positioned_slices_by_the_fitted_offset():
    mirror, abba = _mirror()
    state = _stack()
    for index, s in enumerate(state.slices):
        s.position_mm = 2.0 + index

    mirror.on_write(state)

    moved = {slice_obj._name: z for slice_obj, z in abba.mp.moves}
    # fallback offset (1.0, 1.0): z_abba = position_mm + 1.0
    assert moved == {"a.png": 3.0, "b.png": 4.0, "c.png": 5.0}


def test_on_write_only_moves_a_slice_whose_position_or_order_changed():
    mirror, abba = _mirror()
    state = _stack()
    state.slices[0].position_mm = 2.0
    state.slices[1].position_mm = 3.0
    state.slices[2].position_mm = 4.0
    mirror.on_write(state)
    assert len(abba.mp.moves) == 3

    abba.mp.moves.clear()
    state.slices[1].position_mm = 3.5  # only this one changes
    mirror.on_write(state)

    assert [slice_obj._name for slice_obj, _ in abba.mp.moves] == ["b.png"]


# --- provisional order (no positions yet) -----------------------------------


def test_on_write_places_an_unpositioned_stack_by_provisional_index():
    mirror, abba = _mirror()
    state = _stack()  # interval_mm = 0.2, no positions

    mirror.on_write(state)

    moved = {slice_obj._name: z for slice_obj, z in abba.mp.moves}
    anchor = mirror.to_abba_z(0.0)
    assert moved == {
        "a.png": anchor,
        "b.png": anchor + 0.2,
        "c.png": anchor + 0.4,
    }


def test_a_real_position_wins_over_the_provisional_slot():
    mirror, abba = _mirror()
    state = _stack()
    mirror.on_write(state)  # provisional pass

    abba.mp.moves.clear()
    state.slices[1].position_mm = 9.0
    mirror.on_write(state)

    assert [(s._name, z) for s, z in abba.mp.moves] == [("b.png", 10.0)]


# --- flip / rotation idempotence --------------------------------------------


def test_reapplying_the_same_flip_never_accumulates():
    mirror, _abba = _mirror()
    slice_obj = mirror._slice_by_id["a.png"]

    mirror._push_pretransform("a.png", slice_obj, flip=True, rotation_deg=90)
    first = slice_obj.getTransformSourceOrigin()
    mirror._push_pretransform("a.png", slice_obj, flip=True, rotation_deg=90)
    second = slice_obj.getTransformSourceOrigin()

    assert first == second  # same ops, not doubled
    assert len(second.ops) == 2  # one rotate for the turn, one for the flip
    assert len(slice_obj.pretransform_history) == 2  # both calls did write


def test_pretransform_turns_before_it_flips_like_the_renderer():
    """linear/render.py rotates first, then flips left-right; ImgLib2 ops act
    after the transform built so far, so the turn must be pushed first."""
    mirror, _abba = _mirror()
    slice_obj = mirror._slice_by_id["a.png"]
    mirror._push_pretransform("a.png", slice_obj, flip=True, rotation_deg=90)
    ops = slice_obj.getTransformSourceOrigin().ops
    assert [op[1] for op in ops] == [QUARTER_TURN_AXIS, FLIP_ROTATION_AXIS]
    assert ops[0][2] == pytest.approx(math.radians(90 * QUARTER_TURN_SIGN))
    assert ops[1][2] == pytest.approx(math.pi)


def test_on_write_pushes_the_pretransform_only_when_flip_or_rotation_changes():
    mirror, _abba = _mirror()
    state = _stack()
    state.slices[0].flip = True
    mirror.on_write(state)
    slice_obj = mirror._slice_by_id["a.png"]
    assert len(slice_obj.pretransform_history) == 1

    mirror.on_write(state)  # unchanged: no second push
    assert len(slice_obj.pretransform_history) == 1


# --- in-plane affine: replace, not stack; undo deletes ----------------------


def _physical(rotation_deg: float = 5.0) -> dict:
    return {
        "kind": "interactive",
        "params": [1, 0, 0, 0, 1, 0],
        "physical": {
            "rotation_deg": rotation_deg,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "shear": 0.0,
            "translate_x_mm": 0.1,
            "translate_y_mm": -0.2,
            "pivot": [0.5, 0.5],
        },
    }


def _stub_affine_seams(mirror: AbbaStackMirror) -> list[str]:
    """Replace the two Java-object-building seams with a call log; the
    bookkeeping in ``_sync_affine`` (has_affine, replace order) is what is
    under test, not the exact Java call."""
    calls: list[str] = []
    mirror._register_affine = lambda slice_obj, physical: calls.append(  # type: ignore[method-assign]
        f"register:{slice_obj._name}:{physical['rotation_deg']}"
    )
    mirror._delete_last_registration = lambda slice_obj: calls.append(  # type: ignore[method-assign]
        f"delete:{slice_obj._name}"
    )
    return calls


def test_on_write_registers_a_new_affine_once():
    mirror, _abba = _mirror()
    calls = _stub_affine_seams(mirror)
    state = _stack()
    state.slices[0].transform = _physical()

    mirror.on_write(state)

    assert calls == ["register:a.png:5.0"]
    assert mirror._has_affine["a.png"] is True


def test_on_write_replaces_rather_than_stacks_a_revised_affine():
    mirror, _abba = _mirror()
    calls = _stub_affine_seams(mirror)
    state = _stack()
    state.slices[0].transform = _physical(rotation_deg=5.0)
    mirror.on_write(state)

    state.slices[0].transform = _physical(rotation_deg=12.0)
    mirror.on_write(state)

    assert calls == [
        "register:a.png:5.0",
        "delete:a.png",
        "register:a.png:12.0",
    ]
    assert mirror._has_affine["a.png"] is True


def test_undoing_the_affine_deletes_it_and_never_touches_a_slice_without_one():
    mirror, _abba = _mirror()
    calls = _stub_affine_seams(mirror)
    state = _stack()
    state.slices[0].transform = _physical()
    mirror.on_write(state)

    # the tool's undo puts the state back to no transform on that slice
    state.slices[0].transform = None
    mirror.on_write(state)

    assert calls == ["register:a.png:5.0", "delete:a.png"]
    assert mirror._has_affine["a.png"] is False

    # a slice that never had one is left alone
    mirror.on_write(state)
    assert calls == ["register:a.png:5.0", "delete:a.png"]


# --- cutting angles ----------------------------------------------------------


def test_on_write_pushes_cutting_angles_to_the_resliced_atlas():
    mirror, abba = _mirror()
    state = _stack()
    state.cutting_angles_deg = {"pitch": 4.0, "yaw": -2.0}

    mirror.on_write(state)

    resliced = abba.mp.getReslicedAtlas()
    assert resliced.rotate_x is not None and resliced.rotate_y is not None


# --- finish ------------------------------------------------------------------


def test_finish_syncs_once_more_and_saves_state():
    mirror, abba = _mirror()
    state = _stack()
    state.slices[0].position_mm = 2.0

    mirror.finish(state, save_state="/tmp/out.abba")

    assert abba.mp.moves  # the final sync ran
    assert abba.state_saves == ["/tmp/out.abba"]


def test_a_failing_observer_call_never_raises():
    mirror, abba = _mirror()

    def boom(*_a, **_k):
        raise RuntimeError("ABBA hiccup")

    mirror._register_affine = boom  # type: ignore[method-assign]
    state = _stack()
    state.slices[0].transform = _physical()

    mirror.on_write(state)  # must not raise


def test_attach_mapping_does_not_import_or_guess_from_slice_order():
    abba = FakeAbba()
    first, second = FakeSlice("first", 5), FakeSlice("second", 2)
    abba.mp._slices = [first, second]
    mirror = AbbaStackMirror(
        abba,
        ["/stack/a.png", "/stack/b.png"],
        slice_by_id={"a.png": first, "b.png": second},
    )
    assert abba.mp.getSlices() == [first, second]
    assert mirror._slice_by_id == {"a.png": first, "b.png": second}


@pytest.mark.parametrize("error", ["names", "foreign", "duplicate"])
def test_attach_mapping_rejects_ambiguous_or_foreign_slices(error):
    abba = FakeAbba()
    first, second = FakeSlice("first", 5), FakeSlice("second", 2)
    abba.mp._slices = [first, second]
    mapping = {"a.png": first, "b.png": second}
    if error == "names":
        mapping = {"other.png": first}
    elif error == "foreign":
        mapping["b.png"] = FakeSlice("foreign", 2)
    else:
        mapping["b.png"] = first
    with pytest.raises(ValueError):
        AbbaStackMirror(abba, ["/stack/a.png", "/stack/b.png"], slice_by_id=mapping)


def test_attached_position_job_preserves_orientation_registration_and_angles(monkeypatch):
    abba = FakeAbba()
    sl = FakeSlice("existing", 5)
    sl._pretransform = FakeAffineTransform3D([("existing",)])
    abba.mp._slices = [sl]
    abba.mp._resliced.rotate_x = 0.4
    mirror = AbbaStackMirror(
        abba,
        ["/stack/a.png"],
        slice_by_id={"a.png": sl},
        tasks=["position"],
        sync_angles=False,
    )
    calls = []
    monkeypatch.setattr(mirror, "_sync_affine", lambda *args: calls.append(args))
    state = _stack(1)
    state.slices[0].position_mm = 3
    state.slices[0].flip = True
    state.slices[0].transform = _physical()
    mirror.on_write(state)
    assert abba.mp.moves == [(sl, 4)]
    assert sl.pretransform_history == []
    assert sl._pretransform.ops == [("existing",)]
    assert calls == []
    assert abba.mp._resliced.rotate_x == 0.4


def test_existing_runner_seeds_host_state_without_reset_or_resume(tmp_path, monkeypatch):
    from PIL import Image

    from langslice.integrations.abba_linear import run_existing_in_abba
    from langslice.linear import engine
    from langslice.linear.job import apply_host_inputs
    from langslice.linear.spec import JobSpec

    abba = FakeAbba()
    first, second = FakeSlice("first", 6), FakeSlice("second", 3)
    first._pretransform = FakeAffineTransform3D([("user",)])
    abba.mp._slices = [first, second]
    for name in ("a.png", "b.png"):
        Image.new("RGB", (10, 10)).save(tmp_path / name)
    seen = []
    displayed = []

    async def fake_run(spec, *, emit, on_write):
        seen.append(spec)
        state = _stack(2)
        apply_host_inputs(state, spec)
        on_write(state)
        assert abba.mp.moves == []
        assert first.pretransform_history == []
        state.slices[0].position_mm = 4
        on_write(state)
        return state

    monkeypatch.setattr(engine, "run", fake_run)
    original = JobSpec(str(tmp_path), tasks=["position"], inputs={"fact": "retained"})
    state = run_existing_in_abba(
        abba,
        original,
        slice_by_id={"a.png": first, "b.png": second},
        save_state="result.abba",
        emit=lambda _: None,
        on_write=lambda state, mapping: displayed.append((state.to_dict(), mapping)),
    )
    assert original.resume is True
    assert original.inputs == {"fact": "retained"}
    assert seen[0].resume is False
    assert seen[0].inputs["order"] == ["b.png", "a.png"]
    assert seen[0].inputs["positions"] == {"a.png": 5, "b.png": 2}
    assert seen[0].inputs["angles"] == {"pitch": 0, "yaw": 0}
    assert abba.mp.moves == [(first, 5)]
    assert first._pretransform.ops == [("user",)]
    assert abba.mp._resliced.rotate_x is None
    assert abba.state_saves == ["result.abba"]
    assert (tmp_path / "abba_run.json").is_file()
    assert state.slices[0].position_mm == 4
    assert len(displayed) == 2
    assert displayed[0][0]["slices"][0]["position_mm"] == 5
    assert displayed[1][0]["slices"][0]["position_mm"] == 4
    assert displayed[1][1] == {"a.png": first, "b.png": second}


def test_existing_runner_refuses_orientation_on_registered_slice(tmp_path, monkeypatch):
    from langslice.integrations.abba_linear import run_existing_in_abba
    from langslice.linear.spec import JobSpec

    abba = FakeAbba()
    sl = FakeSlice("existing", 4)
    abba.mp._slices = [sl]
    monkeypatch.setattr(sl, "getNumberOfRegistrations", lambda: 1)
    with pytest.raises(ValueError, match="Turn off ordering"):
        run_existing_in_abba(abba, JobSpec(str(tmp_path)), emit=lambda _: None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("unsupported", ["atlas", "plane", "angle"])
def test_existing_runner_rejects_unverified_atlas_geometry(tmp_path, unsupported):
    from langslice.integrations.abba_linear import run_existing_in_abba
    from langslice.linear.spec import JobSpec

    abba = FakeAbba()
    abba.mp._slices = [FakeSlice("existing", 4)]
    spec = JobSpec(str(tmp_path))
    if unsupported == "atlas":
        spec.atlas = "whs_sd_rat_39um"
    elif unsupported == "plane":
        spec.plane = "sagittal"
    else:
        abba.mp._resliced.rotate_x = 0.01
    with pytest.raises(ValueError):
        run_existing_in_abba(abba, spec, emit=lambda _: None)
    assert list(tmp_path.iterdir()) == []


def test_snapshot_mirror_uses_exact_matrix_and_replaces_only_owned_registration(monkeypatch):
    import numpy as np

    mirror, abba = _mirror()
    mirror._snapshot_geometry["a.png"] = ((800, 600), 25.0)
    mirror._rotation_by_id["a.png"] = 90
    writes, deletes = [], []
    monkeypatch.setattr(mirror, "_register_matrix", lambda sl, matrix: writes.append(matrix))
    monkeypatch.setattr(mirror, "_delete_last_registration", lambda sl: deletes.append(sl))
    transform = {"params": [1, 0.2, 0.1, 0, 1, -0.2]}
    sl = abba.mp.getSlices()[0]
    mirror._sync_affine("a.png", sl, transform)
    assert deletes == []
    # Oriented physical frame is 15 mm wide, 20 mm high; includes shear
    # and the centre correction, without requiring rounded physical knobs.
    np.testing.assert_allclose(writes[0], [[1, 0.15, 0, 3], [0, 1, 0, -4], [0, 0, 1, 0]])
    mirror._sync_affine("a.png", sl, transform)
    assert deletes == [sl]
    mirror._sync_affine("a.png", sl, None)
    assert deletes == [sl, sl]
    assert mirror._has_affine["a.png"] is False


def test_spline_replaces_owned_affine_and_undo_restores_affine(monkeypatch):
    import numpy as np

    mirror, abba = _mirror()
    mirror._snapshot_geometry["a.png"] = ((800, 600), 25.0)
    calls = []
    prepared = object()
    monkeypatch.setattr(mirror, "_prepare_spline_registration", lambda src, tgt: prepared)
    monkeypatch.setattr(mirror, "_append_registration", lambda sl, reg: calls.append(reg))
    monkeypatch.setattr(mirror, "_register_matrix", lambda sl, mat: calls.append("affine"))
    monkeypatch.setattr(mirror, "_delete_last_registration", lambda sl: calls.append("delete"))
    sl = abba.mp.getSlices()[0]
    affine = {"params": [1, 0, .05, 0, 1, 0]}
    spline = {**affine, "spline": {
        "source": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "target": [[.05, 0], [1.05, 0], [.05, 1], [1.1, 1]],
        "extent_mm": [20, 15],
    }}
    mirror._sync_affine("a.png", sl, affine)
    mirror._sync_affine("a.png", sl, spline)
    mirror._sync_affine("a.png", sl, spline)
    mirror._sync_affine("a.png", sl, affine)
    assert calls == ["affine", "delete", prepared, "delete", prepared, "delete", "affine"]
    assert mirror._has_affine["a.png"]
    # The params remain metadata: no additional affine was pushed for the spline.
    assert np.count_nonzero([x == "affine" for x in calls]) == 2


def test_spline_requires_snapshot_geometry_before_removing_old_step(monkeypatch):
    mirror, abba = _mirror()
    mirror._has_affine["a.png"] = True
    calls = []
    monkeypatch.setattr(mirror, "_delete_last_registration", lambda sl: calls.append(sl))
    with pytest.raises(ValueError, match="calibrated ABBA snapshots"):
        mirror._sync_affine("a.png", abba.mp.getSlices()[0], {"spline": {}})
    assert calls == []
    assert mirror._has_affine["a.png"]


def test_invalid_native_spline_does_not_remove_previous_registration(monkeypatch):
    mirror, abba = _mirror()
    mirror._snapshot_geometry["a.png"] = ((800, 600), 25.)
    mirror._has_affine["a.png"] = True
    calls = []
    monkeypatch.setattr(mirror, "_delete_last_registration", lambda sl: calls.append(sl))
    with pytest.raises(ValueError, match="physical extent"):
        mirror._sync_affine("a.png", abba.mp.getSlices()[0], {"spline": {"extent_mm": [1, 1]}})
    assert calls == []


def test_failed_registration_sync_records_actionable_error_and_retries(monkeypatch):
    mirror, abba = _mirror()
    state = _stack()
    state.slices[0].transform = {"kind": "interactive", "spline": {}}
    mirror.on_write(state)
    assert "calibrated ABBA snapshots" in mirror.sync_errors["a.png"]
    calls = []
    monkeypatch.setattr(mirror, "_sync_affine", lambda *args: calls.append(args))
    mirror.on_write(state)
    assert len(calls) == 1
    assert mirror.sync_errors == {}
