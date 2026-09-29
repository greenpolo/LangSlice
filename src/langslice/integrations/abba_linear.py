"""Live ABBA mirror for the linear agent — watch it work in BigDataViewer.

:class:`AbbaStackMirror` attaches to :func:`langslice.linear.engine.run`'s
``on_write`` hook (every write funnels through
:func:`langslice.linear.checkpoint.save_checkpoint`, which is the one place
every tool's result passes through). On each write it diffs the new
:class:`~langslice.linear.state.StackState` against the last one it saw and
pushes only what changed into a running ``abba-python`` session:

- order/position -> ``MultiSlicePositioner.moveSlice`` (grouped between two
  ``MarkActionSequenceBatchAction`` calls so one agent write is one ABBA undo
  step);
- flip / quarter-turn (``SliceState.flip``/``rotation_deg``) -> the slice's
  pre-transform (``SliceSources.transformSourceOrigin``), rebuilt fresh from
  the slice's captured import-time base transform every time, so re-applying
  the same correction never accumulates;
- the in-plane affine (``SliceState.transform.physical``) -> a registration
  step on the slice's stack (``RegisterSliceAction`` with an
  ``AffineRegistration``), replaced rather than stacked when the agent
  revises it (``DeleteLastRegistrationAction`` then a fresh push);
- the stack's cutting angles -> ``ReslicedAtlas.setRotateX``/``setRotateY``.

The agent itself is unchanged: it still renders its own BrainGlobe pictures
and reasons over its own toolbox exactly as it does headless. ABBA is
display, plus the final home of the result (``AbbaStackMirror.finish`` can
write an ``.abba`` state file). ``damaged`` has no ABBA equivalent and is not
mirrored.

The module-level ``*_SIGN``/``*_AXIS`` constants carry the in-plane sign
conventions. The flip axis, quarter-turn sign, in-plane rotation sign,
translation signs and scale axis were MEASURED headlessly on 2026-09-10
(headless probe registrations, scripts kept outside the repo):
a probe registration reads back what ABBA itself resamples after each write,
and ABBA's own coordinate channels say ML DECREASES with screen x while DV
increases with screen y — so the probe grid (ML right) is a left-right mirror
of the screen, and the two rotation signs come out NEGATIVE (ImgLib2's
``rotate`` is counter-clockwise in a y-up frame, i.e. clockwise on a y-down
screen, where LangSlice's ``rotation_deg`` is counter-clockwise). The
pitch/yaw mapping onto ``ReslicedAtlas.setRotateX/Y`` was measured on
2026-09-22: pitch flips sign, yaw does not, magnitudes agree.

This module imports JPype/scyjava/abba_python lazily, inside the functions
that need them, exactly like :mod:`langslice.integrations.abba` — so it
imports cleanly (and its diff logic is unit-testable) in the plain ``.venv``,
which does not have JPype installed at all.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langslice.linear.spec import JobSpec
    from langslice.linear.state import StackState

logger = logging.getLogger(__name__)

#: Display name of the in-plane-affine registration step this mirror pushes
#: (RegisterSliceAction/AffineRegistration), distinct from
#: ``langslice.integrations.abba.REGISTRATION_NAME`` (the *plugin type* name
#: the nonlinear adapter registers once per session).
REGISTRATION_STEP_NAME = "LangSlice affine"

#: Anchor interval (mm) used to space an unpositioned stack along ABBA's
#: slicing axis so it stays visible before any real position lands.
PROVISIONAL_INTERVAL_MM = 0.1

# --- sign/axis conventions --------------------------------------------------
# VERIFIED 2026-09-10 (spike8 + spike9, headless probe registrations on
# LSD_910/M01 slice 14; NCC of ABBA's resampled slice against LangSlice's own
# physical_affine_matrix warp of the baseline, read back in screen terms):
#   flip           -> left-right mirror on screen   (NCC 0.83 vs -0.07 up-down)
#   quarter-turn   -> needs sign -1                 (0.89 vs -0.07)
#   rotation_deg   -> needs sign -1                 (0.84 vs -0.16)
#   translate_x/y  -> +3 mm moved +3.00 / +2.60 mm  (signs +1)
#   scale_x        -> scales screen x               (0.90 vs -0.43 for y)
# pitch/yaw <-> setRotateX/setRotateY: rotateX is pitch and rotateY is yaw,
# BOTH with the opposite sign, in radians of the same angle LangSlice uses.
# Pinned 2026-09-29 to ABBA's own exported chains (SliceBench ABBA ingest:
# stage 0 of every LSD_910 QuPath ABBA-Transform export; M12's plane normal
# matches pitch=-rx, yaw=-ry to 0.0000 deg, yaw=+ry is 5.9 deg off) and to
# screen sides: in ABBA a +rotateY cut puts the screen-LEFT half more
# posterior; LangSlice's coronal display has the BrainGlobe ML index running
# rightward, so that is yaw < 0. Spike 12 (2026-09-22) read yaw = +rotateY
# because it took ABBA's ML coordinate channel as BrainGlobe ML; that channel
# DECREASES with screen x (legacy Allen map: ML = 11.4 - 0.01*k, "Left" where
# it is < 5.7), i.e. it is mirrored against the display. The Allen volume is
# exactly left-right symmetric, so its label score could not see the mirror.

#: SliceSources.rotateSourceOrigin axis for SliceState.flip (mirror
#: left-right): rotating pi about Y. SliceSources.java:547-551.
FLIP_ROTATION_AXIS = 1
#: ... axis for SliceState.rotation_deg (the quarter-turn correction): Z.
QUARTER_TURN_AXIS = 2
#: Sign applied to SliceState.rotation_deg (CCW on screen) before ImgLib2's
#: Z rotation, which is clockwise on ABBA's y-down screen. Measured.
QUARTER_TURN_SIGN = -1.0
#: Sign applied to the in-plane affine's rotation_deg (LangSlice: CCW on
#: screen, affine.physical_affine_matrix) before ImgLib2's Z rotation. Same
#: reason as QUARTER_TURN_SIGN. Measured.
INPLANE_ROTATION_SIGN = -1.0
#: Sign applied to translate_y_mm (LangSlice: +y is DOWN on screen) before
#: ImgLib2's Y translation in the aligner frame. ABBA's DV also increases
#: down the screen, so no flip. Measured.
INPLANE_TRANSLATE_Y_SIGN = 1.0
#: Which of ReslicedAtlas.setRotateX/setRotateY is pitch vs yaw, and sign.
#: ReslicedAtlas.java:333,356. pitch_deg = -deg(rotateX), yaw_deg =
#: -deg(rotateY); see the note above and tests/test_integrations_abba_math.py.
PITCH_TO_ROTATE_X_SIGN = -1.0
YAW_TO_ROTATE_Y_SIGN = -1.0


# ---------------------------------------------------------------------------
# Coordinates: BrainGlobe AP mm <-> ABBA slicing-axis mm
# ---------------------------------------------------------------------------


def to_abba_z(position_mm: float, a: float, b: float) -> float:
    """ABBA slicing-axis mm for a BrainGlobe position, given a fit ``(a, b)``
    from :func:`measure_axis_offset` (``z_abba = a * position_mm + b``)."""
    return a * position_mm + b


def from_abba_z(z_abba: float, a: float, b: float) -> float:
    """Inverse of :func:`to_abba_z`."""
    return (z_abba - b) / a


def measure_axis_offset(
    abba: Any, *, z1_mm: float = 3.0, z2_mm: float = 9.0
) -> tuple[float, float]:
    """Fit ABBA's slicing-axis mm to BrainGlobe AP mm: ``z_abba = a*ap+b``.

    UNVERIFIED-then-fixed once, live: the first version of this sampled
    ``mp.getReslicedAtlas().extendedSlicedSources`` directly at world points
    ``(0, 0, z1)``/``(0, 0, z2)``. A live diagnostic
    (probe script kept outside the repo) showed that source's own
    Z axis is a static ~0.3 mm margin anchored at a FIXED reference plane
    (ABBA's own z=0), not a sweep across the atlas's whole AP range — both
    probes landed out of bounds and read back an identical background value.
    ``mp.getReslicedAtlas()`` is the shared cross-section VIEW, not a
    general 3D sampler.

    What actually varies the coordinate channel with z, proven by a live
    probe registration (script kept outside the repo): ``registerSelectedSlices``
    always composes a ``SourcesZOffset(slice)`` onto the fixed-image
    preprocessor (``MultiSlicePositioner.registerSelectedSlices``, and it is
    exactly what the nonlinear adapter in ``langslice.integrations.abba``
    relies on). So this moves the aligner's first slice to *z1_mm* and
    *z2_mm* in turn and, at each, runs one throwaway registration with a
    probe plugin that reads the fixed image's AP channel at its centre pixel
    (:func:`langslice.integrations.abba._imageplus_to_numpy`) — the same
    per-pixel AP mm the nonlinear path samples the atlas at. The probe
    registration is removed and the slice's original position restored
    afterward.

    Never hardcode the measured offset (~0.985-0.99, not exactly 1.0) as
    ``1.0``; call this once per session instead. Falls back to ``(1.0, 1.0)``
    with a loud warning if sampling fails for any reason — a JPype/API
    mismatch must stop the mirror from being CALIBRATED, never from running.
    """
    try:
        from jpype import JImplements, JOverride  # pyright: ignore[reportMissingImports]
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        from langslice.integrations.abba import COORD_CHANNELS, _imageplus_to_numpy

        slices = list(abba.mp.getSlices())
        if not slices:
            raise RuntimeError("no slice imported yet to probe the offset with")
        probe = slices[0]
        original_z = float(probe.getSlicingAxisPosition())

        MultiSlicePositioner = jimport("ch.epfl.biop.atlas.aligner.MultiSlicePositioner")
        SimpleRegistrationWrapper = jimport(
            "ch.epfl.biop.registration.plugin.SimpleRegistrationWrapper"
        )
        SourcesChannelsSelect = jimport(
            "ch.epfl.biop.sourceandconverter.processor.SourcesChannelsSelect"
        )
        DeleteLastRegistrationAction = jimport(
            "ch.epfl.biop.atlas.aligner.DeleteLastRegistrationAction"
        )
        AffineTransform3D = jimport("net.imglib2.realtransform.AffineTransform3D")
        IJ = jimport("ij.IJ")
        HashMap = jimport("java.util.HashMap")
        Supplier = jimport("java.util.function.Supplier")

        plugin_name = "LangSlice-AxisOffsetProbe"
        captured: dict[str, float] = {}

        @JImplements("ch.epfl.biop.registration.plugin.SimpleRegistrationPlugin")
        class _ProbePlugin:
            @JOverride
            def getVoxelSizeInMicron(self):
                return 40.0

            @JOverride
            def setRegistrationParameters(self, parameters):
                pass

            @JOverride
            def register(self, fixed, moving, fixedMask, movingMask):
                coords = _imageplus_to_numpy(fixed, IJ)  # (channels, H, W)
                h, w = coords.shape[-2], coords.shape[-1]
                captured["ap"] = float(coords[0, h // 2, w // 2])
                return AffineTransform3D()

        @JImplements(Supplier)
        class _ProbeSupplier:
            @JOverride
            def get(self):
                return SimpleRegistrationWrapper(plugin_name, _ProbePlugin())

        MultiSlicePositioner.registerRegistrationPlugin(plugin_name, _ProbeSupplier())
        previously_selected = list(abba.mp.getSelectedSlices())
        abba.mp.deselectSlice(abba.mp.getSlices())
        abba.mp.selectSlice(probe)

        def ap_mm_at(z_abba: float) -> float:
            abba.mp.moveSlice(probe, float(z_abba))
            abba.wait_for_end_of_tasks()
            captured.clear()
            abba.mp.registerSelectedSlices(
                plugin_name,
                SourcesChannelsSelect(list(COORD_CHANNELS)),
                SourcesChannelsSelect(0),
                HashMap(),
            )
            abba.wait_for_end_of_tasks()
            DeleteLastRegistrationAction(abba.mp, probe).runRequest()
            if "ap" not in captured:
                raise RuntimeError(f"probe registration never ran at z={z_abba}")
            return captured["ap"]

        try:
            ap1, ap2 = ap_mm_at(z1_mm), ap_mm_at(z2_mm)
        finally:
            abba.mp.moveSlice(probe, original_z)
            abba.mp.deselectSlice(abba.mp.getSlices())
            if previously_selected:
                abba.mp.selectSlice(previously_selected)
            abba.wait_for_end_of_tasks()

        if ap1 == ap2:
            raise ValueError(f"AP coordinate channel is flat between z={z1_mm} and z={z2_mm}")
        a = (z1_mm - z2_mm) / (ap1 - ap2)
        b = z1_mm - a * ap1
        if not (0.5 < abs(a) < 2.0):
            logger.warning(
                "measure_axis_offset: |a|=%.3f is far from 1 (expected the two "
                "systems to differ only by an offset and maybe a sign)",
                a,
            )
        logger.info("measure_axis_offset: z_abba = %.4f * ap_mm + %.4f", a, b)
        return a, b
    except Exception:
        logger.warning(
            "measure_axis_offset failed; falling back to (1.0, 1.0) -- ABBA z "
            "and BrainGlobe AP mm will be treated as numerically identical, "
            "which the measured offset (~0.985-0.99) says is only approximate",
            exc_info=True,
        )
        return 1.0, 1.0


# ---------------------------------------------------------------------------
# The mirror
# ---------------------------------------------------------------------------


class AbbaStackMirror:
    """Push every linear-agent write into a live ABBA session.

    ``AbbaStackMirror(abba, image_paths)`` imports a fresh stack. Pass an
    explicit ``slice_by_id`` to attach without importing; every filename
    must map to one distinct slice already in that ABBA session. ``tasks``
    and ``sync_angles`` restrict which properties the mirror may change.
    Attach ``mirror.on_write`` as
    :func:`langslice.linear.engine.run`'s ``on_write``; call
    :meth:`finish` once the session ends (submitted or not).
    """

    def __init__(
        self,
        abba: Any,
        image_paths: list[str],
        *,
        z_increment_mm: float = 0.1,
        slice_by_id: dict[str, Any] | None = None,
        tasks: list[str] | None = None,
        sync_angles: bool = True,
    ) -> None:
        self.abba = abba
        self.mp = abba.mp
        self._last_state: dict[str, Any] | None = None
        self._has_affine: dict[str, bool] = {}
        self._slice_by_id: dict[str, Any] = {}
        self._base_transform: dict[str, Any] = {}
        self._tasks = frozenset(tasks) if tasks is not None else None
        self._sync_angles = sync_angles
        self._snapshot_geometry: dict[str, tuple[tuple[int, int], float]] = {}
        self._rotation_by_id: dict[str, int] = {}
        self.sync_errors: dict[str, str] = {}

        if slice_by_id is not None:
            names = [os.path.basename(path) for path in image_paths]
            if len(set(names)) != len(names) or set(names) != set(slice_by_id):
                raise ValueError("Image filenames must match the ABBA slice mapping exactly")
            session_slices = list(self.mp.getSlices())
            mapped = list(slice_by_id.values())
            if any(sl not in session_slices for sl in mapped):
                raise ValueError("Mapped slices must belong to this ABBA session")
            if any(sl in mapped[:index] for index, sl in enumerate(mapped)):
                raise ValueError("Each image must map to a different ABBA slice")
            self._slice_by_id = dict(slice_by_id)
            for name, slice_obj in self._slice_by_id.items():
                self._has_affine[name] = False
                self._base_transform[name] = slice_obj.getTransformSourceOrigin().copy()
            self._a, self._b = measure_axis_offset(abba)
            return

        # abba_python's own import_from_files already blocks and calls
        # .get() internally (its docstring says it returns a Future, but the
        # actual return value -- verified live -- is the resolved
        # CommandModule; calling .get() on it again raises AttributeError).
        abba.import_from_files(
            filepaths=list(image_paths), z_location=0.0, z_increment=z_increment_mm
        )
        abba.wait_for_end_of_tasks()

        # Import order == increasing z (we just set it that way); sort by
        # the axis position ABBA itself derived, per the task brief, rather
        # than trust getSlices()'s own iteration order.
        slices = sorted(list(abba.mp.getSlices()), key=lambda s: float(s.getSlicingAxisPosition()))
        if len(slices) != len(image_paths):
            raise ValueError(
                "Imported files do not match the ABBA stack; use an explicit "
                "slice mapping when attaching to an existing session"
            )
        for path, slice_obj in zip(image_paths, slices, strict=False):
            slice_id = os.path.basename(path)
            self._slice_by_id[slice_id] = slice_obj
            self._has_affine[slice_id] = False
            try:
                self._base_transform[slice_id] = slice_obj.getTransformSourceOrigin().copy()
            except Exception:
                logger.warning(
                    "AbbaStackMirror: could not read %s's base pre-transform",
                    slice_id,
                    exc_info=True,
                )
            try:
                reported = str(slice_obj.getName())
                stem = os.path.splitext(slice_id)[0]
                if stem not in reported and slice_id not in reported:
                    logger.debug(
                        "AbbaStackMirror: slice.getName() (%r) does not "
                        "obviously match %r; relying on import order, not "
                        "name, for this mapping",
                        reported,
                        slice_id,
                    )
            except Exception:
                logger.debug("AbbaStackMirror: slice.getName() unavailable", exc_info=True)

        self._a, self._b = measure_axis_offset(abba)

    # --- coordinates -------------------------------------------------

    def to_abba_z(self, position_mm: float) -> float:
        return to_abba_z(position_mm, self._a, self._b)

    def from_abba_z(self, z_abba: float) -> float:
        return from_abba_z(z_abba, self._a, self._b)

    # --- the observer --------------------------------------------------

    def on_write(self, state: StackState) -> None:
        """The ``on_write`` hook: never raises, whatever ABBA does."""
        try:
            self._on_write_unsafe(state)
        except Exception:
            logger.warning(
                "AbbaStackMirror.on_write failed; the ABBA display may now "
                "be stale until the next write",
                exc_info=True,
            )

    def _on_write_unsafe(self, state: StackState) -> None:
        current = state.to_dict()
        previous = self._last_state
        prev_rows = {row["id"]: row for row in (previous or {}).get("slices", [])}

        self._mark_batch_boundary()
        for row in current.get("slices", []):
            self._sync_slice(state, row, prev_rows.get(row["id"]))
        try:
            if self._sync_angles:
                self._sync_cutting_angles(previous, current)
        except Exception:
            logger.warning("AbbaStackMirror: cutting-angle sync failed", exc_info=True)
        self._mark_batch_boundary()

        self._last_state = current

    def finish(self, state: StackState, *, save_state: str | None = None) -> None:
        """Final sync, then optionally write an ``.abba`` state file."""
        self.on_write(state)
        if save_state:
            try:
                self.abba.state_save(save_state)
                self.abba.wait_for_end_of_tasks()
            except Exception:
                logger.warning("AbbaStackMirror: state_save(%s) failed", save_state, exc_info=True)

    # --- per-slice sync --------------------------------------------------

    def _sync_slice(
        self, state: StackState, row: dict[str, Any], prev_row: dict[str, Any] | None
    ) -> None:
        slice_id = row["id"]
        self._rotation_by_id[slice_id] = int(row.get("rotation_deg") or 0)
        slice_obj = self._slice_by_id.get(slice_id)
        if slice_obj is None:
            logger.warning("AbbaStackMirror: no ABBA slice for %r; skipping", slice_id)
            return

        position_enabled = self._tasks is None or bool({"position", "reorder"} & self._tasks)
        if position_enabled and (
            prev_row is None
            or (
                row.get("position_mm") != prev_row.get("position_mm")
                or row.get("index_corrected") != prev_row.get("index_corrected")
            )
        ):
            try:
                self.mp.moveSlice(slice_obj, float(self._target_z(state, row)))
            except Exception:
                logger.warning("AbbaStackMirror: moveSlice failed for %s", slice_id, exc_info=True)

        if (self._tasks is None or "reorder" in self._tasks) and (
            prev_row is None
            or (
                row.get("flip") != prev_row.get("flip")
                or row.get("rotation_deg") != prev_row.get("rotation_deg")
            )
        ):
            try:
                self._push_pretransform(
                    slice_id, slice_obj, bool(row.get("flip")), int(row.get("rotation_deg") or 0)
                )
            except Exception:
                logger.warning(
                    "AbbaStackMirror: pre-transform failed for %s", slice_id, exc_info=True
                )

        if (self._tasks is None or "transform" in self._tasks) and (
            prev_row is None or row.get("transform") != prev_row.get("transform")
            or slice_id in self.sync_errors
        ):
            try:
                self._sync_affine(slice_id, slice_obj, row.get("transform"))
                self.sync_errors.pop(slice_id, None)
            except Exception as exc:
                self.sync_errors[slice_id] = str(exc)
                logger.warning(
                    "AbbaStackMirror: registration sync failed for %s",
                    slice_id,
                    exc_info=True,
                )

    def _target_z(self, state: StackState, row: dict[str, Any]) -> float:
        """The z to move a slice to: its real position, or — before one is
        written — a provisional slot by corrected index, so the stack stays
        visible and in order in the atlas."""
        position_mm = row.get("position_mm")
        if position_mm is not None:
            return self.to_abba_z(float(position_mm))
        interval_mm = state.interval_mm or PROVISIONAL_INTERVAL_MM
        anchor = self.to_abba_z(0.0)
        return anchor + int(row.get("index_corrected") or 0) * interval_mm

    def _push_pretransform(
        self, slice_id: str, slice_obj: Any, flip: bool, rotation_deg: int
    ) -> None:
        """Rebuild the pre-transform fresh from the captured import-time
        ``base`` every call, so re-applying the same flip/turn is idempotent
        and never accumulates onto whatever the previous call left behind.

        Order matters: LangSlice renders a section by ROTATING first, then
        flipping left-right (``linear/render.py``), and every ImgLib2
        ``rotate`` call acts AFTER the transform built so far — so the
        quarter-turn is applied before the flip here too. Measured
        2026-09-10 (spike11): the other order lands a flipped, quarter-turned
        section rotated the wrong way."""
        base = self._base_transform.get(slice_id)
        if base is None:
            raise RuntimeError(f"no base pre-transform captured for {slice_id!r}")
        at3d = base.copy()
        if rotation_deg:
            at3d.rotate(QUARTER_TURN_AXIS, math.radians(rotation_deg * QUARTER_TURN_SIGN))
        if flip:
            at3d.rotate(FLIP_ROTATION_AXIS, math.pi)
        slice_obj.transformSourceOrigin(at3d)

    def _sync_affine(self, slice_id: str, slice_obj: Any, transform: dict[str, Any] | None) -> None:
        """Push the agent's in-plane affine as a registration step, replacing
        (never stacking) a LangSlice affine this mirror already pushed."""
        if transform is not None and transform.get("spline") is not None:
            from langslice.integrations.abba_spline import spline_world_landmarks

            geometry = self._snapshot_geometry.get(slice_id)
            if geometry is None:
                raise ValueError(
                    "Landmark warps require calibrated ABBA snapshots. Start the run "
                    "from the LangSlice menu in ABBA so the slice frame is recorded."
                )
            source, target = spline_world_landmarks(
                transform["spline"], size=geometry[0], pixel_size_um=geometry[1],
                rotation_deg=self._rotation_by_id.get(slice_id, 0),
            )
            registration = self._prepare_spline_registration(source, target)
            if self._has_affine.get(slice_id, False):
                self._delete_last_registration(slice_obj)
            self._append_registration(slice_obj, registration)
            self._has_affine[slice_id] = True
            return
        physical = (transform or {}).get("physical") if transform else None
        had = self._has_affine.get(slice_id, False)
        geometry = self._snapshot_geometry.get(slice_id)
        if transform is not None and geometry is not None:
            from langslice.integrations.abba_affine import normalized_to_abba_affine

            size, pixel_size_um = geometry
            matrix = normalized_to_abba_affine(
                transform.get("params"),
                size=size,
                pixel_size_um=pixel_size_um,
                rotation_deg=self._rotation_by_id.get(slice_id, 0),
            )
            if had:
                self._delete_last_registration(slice_obj)
            self._register_matrix(slice_obj, matrix)
            self._has_affine[slice_id] = True
            return
        if physical is None:
            if had:
                self._delete_last_registration(slice_obj)
                self._has_affine[slice_id] = False
            return
        if had:
            self._delete_last_registration(slice_obj)
        self._register_affine(slice_obj, physical)
        self._has_affine[slice_id] = True

    def _sync_cutting_angles(
        self, previous: dict[str, Any] | None, current: dict[str, Any]
    ) -> None:
        angles = current.get("cutting_angles_deg") or {}
        if previous is not None and previous.get("cutting_angles_deg") == angles:
            return
        self._set_cutting_angles(float(angles.get("pitch", 0.0)), float(angles.get("yaw", 0.0)))

    # --- seams: the only calls that build new Java objects from scratch --
    #
    # Everything above calls methods on objects ABBA already gave us (the
    # slice, its pre-transform, ``mp``, the resliced atlas), so it works
    # against a plain-Python fake in tests with no JPype involved at all.
    # These three build a brand new Java object graph and so need a real
    # JVM; tests stub them directly.

    def _mark_batch_boundary(self) -> None:
        """One call opens a batch, the next closes it (MultiSlicePositioner.
        MarkActionSequenceBatchAction.java): everything a write pushes
        between the two undoes as one ABBA step. Never fatal — a session
        with no JVM (or an ABBA version missing the class) just loses the
        batching, not the mirroring."""
        try:
            from scyjava import jimport  # pyright: ignore[reportMissingImports]

            jimport("ch.epfl.biop.atlas.aligner.action.MarkActionSequenceBatchAction")(
                self.mp
            ).runRequest()
        except Exception:
            logger.debug("AbbaStackMirror: undo-batch boundary skipped", exc_info=True)

    def _set_cutting_angles(self, pitch_deg: float, yaw_deg: float) -> None:
        try:
            resliced = self.mp.getReslicedAtlas()
            resliced.setRotateX(math.radians(pitch_deg * PITCH_TO_ROTATE_X_SIGN))
            resliced.setRotateY(math.radians(yaw_deg * YAW_TO_ROTATE_Y_SIGN))
        except Exception:
            logger.warning("AbbaStackMirror: cutting-angle update failed", exc_info=True)

    def _register_affine(self, slice_obj: Any, physical: dict[str, Any]) -> None:
        """Push ``physical`` as a manual affine registration step.

        Built the way ABBA's own QuickNII import builds a manual
        affine registration (``command/ImportSlicesFromQuickNIICommand.
        java:158-190``) — an ``AffineRegistration`` plugin instance carrying
        a JSON-encoded ``AffineTransform3D`` as its "transform" parameter,
        pushed via ``RegisterSliceAction`` under the name
        :data:`REGISTRATION_STEP_NAME`. The transform scales, then rotates
        (about the slice's own centre — never the recorded canvas pivot,
        which is a LangSlice/canvas concept ABBA has no equivalent for), then
        translates by millimetres, in the same centred aligner frame ABBA's
        own "Interactive Transform" tool uses for its rotation/scale/
        translate sliders (``command/SliceAffineTransformCommand.java``).
        Rotation sense and axis directions are measured, not assumed — see
        the sign-constant block at the top of the module. Shear and the
        recorded pivot are not representable here and are dropped.
        """
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        AffineTransform3D = jimport("net.imglib2.realtransform.AffineTransform3D")
        rotation_deg = float(physical.get("rotation_deg", 0.0)) * INPLANE_ROTATION_SIGN
        scale_x = float(physical.get("scale_x", 1.0))
        scale_y = float(physical.get("scale_y", 1.0))
        translate_x_mm = float(physical.get("translate_x_mm", 0.0))
        translate_y_mm = float(physical.get("translate_y_mm", 0.0)) * INPLANE_TRANSLATE_Y_SIGN

        # ImgLib2's scale/rotate/translate each act AFTER the transform built
        # so far, and LangSlice's affine_matrix is translate ∘ rotate ∘ scale
        # (scales "applied before the rotation") — so: scale, rotate, translate.
        # The live M01 run of 2026-09-10 caught the rotate-then-scale order:
        # with unequal scales the two differ, single-knob probes never see it.
        at3d = AffineTransform3D()
        at3d.scale(scale_x, scale_y, 1.0)
        at3d.rotate(QUARTER_TURN_AXIS, math.radians(rotation_deg))
        at3d.translate(translate_x_mm, translate_y_mm, 0.0)
        self._register_java_affine(slice_obj, at3d)

    def _register_matrix(self, slice_obj: Any, matrix: Any) -> None:
        """Append the exact centred snapshot matrix, including pivot and shear."""
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        at3d = jimport("net.imglib2.realtransform.AffineTransform3D")()
        for row in range(3):
            for column in range(4):
                at3d.set(float(matrix[row, column]), row, column)
        self._register_java_affine(slice_obj, at3d)

    def _register_java_affine(self, slice_obj: Any, at3d: Any) -> None:
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        AffineRegistration = jimport(
            "ch.epfl.biop.registration.sourceandconverter.affine.AffineRegistration"
        )
        PluginService = jimport("org.scijava.plugin.PluginService")
        SourcesProcessorHelper = jimport(
            "ch.epfl.biop.sourceandconverter.processor.SourcesProcessorHelper"
        )
        RegisterSliceAction = jimport("ch.epfl.biop.atlas.aligner.RegisterSliceAction")
        MultiSlicePositioner = jimport("ch.epfl.biop.atlas.aligner.MultiSlicePositioner")
        HashMap = jimport("java.util.HashMap")

        ctx = self.abba.ij.context()
        plugin_service = ctx.getService(PluginService.class_)
        registration = plugin_service.getPlugin(AffineRegistration.class_).createInstance()
        registration.setScijavaContext(ctx)

        params = HashMap()
        params.put("transform", AffineRegistration.affineTransform3DToString(at3d))
        params.put("pz", 0)
        registration.setRegistrationParameters(MultiSlicePositioner.convertToString(ctx, params))
        registration.setRegistrationName(REGISTRATION_STEP_NAME)

        RegisterSliceAction(
            self.mp,
            slice_obj,
            registration,
            SourcesProcessorHelper.Identity(),
            SourcesProcessorHelper.Identity(),
        ).runRequest()

    def _prepare_spline_registration(self, source: Any, target: Any) -> Any:
        from langslice.integrations.abba_spline import prepare_spline_registration

        return prepare_spline_registration(self.abba, source, target)

    def _append_registration(self, slice_obj: Any, registration: Any) -> None:
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        helper = jimport("ch.epfl.biop.sourceandconverter.processor.SourcesProcessorHelper")
        jimport("ch.epfl.biop.atlas.aligner.RegisterSliceAction")(
            self.mp, slice_obj, registration, helper.Identity(), helper.Identity(),
        ).runRequest()

    def _delete_last_registration(self, slice_obj: Any) -> None:
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        DeleteLastRegistrationAction = jimport(
            "ch.epfl.biop.atlas.aligner.DeleteLastRegistrationAction"
        )
        DeleteLastRegistrationAction(self.mp, slice_obj).runRequest()


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------


def export_existing_slices(
    abba: Any,
    folder: str,
    *,
    channel: int = 0,
    pixel_size_um: float = 25.0,
    slices: list[Any] | None = None,
) -> dict[str, Any]:
    """Snapshot current registered images on a calibrated, centred XY frame.

    Images remain in *folder* alongside the run checkpoint for reproducibility.
    No ABBA imports, original image edits, or ImageJ image windows are created.
    """
    from pathlib import Path

    from scyjava import jimport  # pyright: ignore[reportMissingImports]

    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Snapshot pixel size must be a positive finite number")
    if channel < 0:
        raise ValueError("Image channel must be zero or greater")
    selected = list(slices) if slices is not None else _selected_or_all_slices(abba)
    if not selected:
        raise ValueError("Import slices into ABBA before starting LangSlice")
    px, py, sx, sy = [float(value) for value in abba.mp.getROI()]
    half_x, half_y = max(abs(px), abs(px + sx)), max(abs(py), abs(py + sy))
    if not all(math.isfinite(value) and value > 0 for value in (half_x, half_y)):
        raise ValueError("ABBA's image region must have positive finite dimensions")
    spacing_mm = pixel_size_um / 1000.0
    # An even number of pixels keeps the physical image centre at ABBA's
    # origin, the pivot used by the existing mirror's affine registration.
    half_x = math.ceil(half_x / spacing_mm) * spacing_mm
    half_y = math.ceil(half_y / spacing_mm) * spacing_mm
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    SliceToImagePlus = jimport("ch.epfl.biop.atlas.aligner.SliceToImagePlus")
    SourcesChannelsSelect = jimport(
        "ch.epfl.biop.sourceandconverter.processor.SourcesChannelsSelect"
    )
    FileSaver = jimport("ij.io.FileSaver")
    mapping: dict[str, Any] = {}
    for index, slice_obj in enumerate(selected):
        if channel >= len(slice_obj.getRegisteredSources()):
            raise ValueError(f"Slice {slice_obj.getName()} has no channel {channel}")
        name = f"section_{index + 1:04d}.tif"
        imp = SliceToImagePlus.export(
            slice_obj,
            SourcesChannelsSelect(channel),
            -half_x,
            -half_y,
            2 * half_x,
            2 * half_y,
            spacing_mm,
            0,
            True,
        )
        try:
            if not FileSaver(imp).saveAsTiff(str(destination / name)):
                raise RuntimeError(f"Could not save ABBA snapshot {name}")
        finally:
            imp.close()
        mapping[name] = slice_obj
    return mapping


def _selected_or_all_slices(abba: Any) -> list[Any]:
    slices = list(abba.mp.getSlices())
    selected = [slice_obj for slice_obj in slices if slice_obj.isSelected()]
    return sorted(selected or slices, key=lambda sl: float(sl.getSlicingAxisPosition()))


def run_existing_in_abba(
    abba: Any,
    spec: JobSpec,
    *,
    slice_by_id: dict[str, Any] | None = None,
    save_state: str | None = None,
    emit: Callable[[str], None] = print,
    channel: int = 0,
    pixel_size_um: float = 25.0,
    on_write: Callable[[StackState, dict[str, Any]], None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    follow_agent: Callable[[], bool] | None = None,
    comparison_factory: Callable[[], Any] | None = None,
) -> StackState:
    """Run against selected ABBA slices (all when none are selected).

    Suitable for a GUI background worker: this never starts or blocks on a
    second GUI. Without an explicit mapping, current registered images are
    staged in a fresh run folder beneath ``spec.image_folder`` (or the system
    temporary directory when empty). With a mapping, that folder must already
    contain exactly the mapped snapshots, calibrated in the ABBA XY frame.

    Existing registrations remain underneath new affine corrections. Reorder
    is refused for registered sections because source-orientation operations
    precede registrations whereas these snapshots represent their result.
    ``on_write`` optionally receives state and the image-to-slice mapping
    after each mirror update, for a host's progress display or follow view.
    """
    import asyncio
    import json
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    from langslice.linear.discovery import discover_slices
    from langslice.linear.engine import run

    if not spec.tasks:
        raise ValueError("Select at least one LangSlice task")
    if spec.plane != "coronal":
        raise ValueError("The live ABBA adapter currently supports coronal sessions")
    if getattr(abba, "z_axis", "AP") != "AP":
        raise ValueError("Start a coronal ABBA session with an AP slicing axis")
    supported_atlases = {
        "Adult Mouse Brain - Allen Brain Atlas V3p1",
        "Adult Mouse Brain - Allen Brain Atlas V3",
    }
    if (
        getattr(abba, "atlas_name", "Adult Mouse Brain - Allen Brain Atlas V3p1")
        not in supported_atlases
    ):
        raise ValueError("The live adapter currently requires ABBA's Allen Mouse V3p1 atlas")
    if spec.atlas not in {"allen_mouse_10um", "allen_mouse_25um", "allen_mouse_50um"}:
        raise ValueError("Choose an Allen Mouse atlas to match this ABBA session")
    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Snapshot pixel size must be a positive finite number")
    resliced = abba.mp.getReslicedAtlas()
    current_angles = (float(resliced.getRotateX()), float(resliced.getRotateY()))
    if any(not math.isfinite(angle) or abs(angle) > 1e-8 for angle in current_angles):
        raise ValueError(
            "Live runs with tilted atlas planes are not verified yet. "
            "Set ABBA's atlas angles to zero before starting LangSlice."
        )
    abba.wait_for_end_of_tasks()
    slices = (
        list(slice_by_id.values()) if slice_by_id is not None else _selected_or_all_slices(abba)
    )
    if not slices:
        raise ValueError("Import slices into ABBA before starting LangSlice")
    if spec.has("reorder") and any(int(sl.getNumberOfRegistrations()) > 0 for sl in slices):
        raise ValueError(
            "Ordering/orientation requires slices without existing registrations. "
            "Turn off ordering to position or align this registered stack."
        )
    run_spec = replace(spec, resume=False, inputs=dict(spec.inputs))
    if slice_by_id is None:
        parent = Path(spec.image_folder).expanduser() if spec.image_folder else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        folder = tempfile.mkdtemp(prefix="langslice-abba-", dir=parent)
        emit(f"[ABBA] Snapshotting {len(slices)} section(s) into {folder}")
        slice_by_id = export_existing_slices(
            abba, folder, channel=channel, pixel_size_um=pixel_size_um, slices=slices
        )
        run_spec = replace(run_spec, image_folder=folder, out=None)
        run_spec.inputs["pixel_size_um"] = pixel_size_um
    paths = discover_slices(os.path.abspath(run_spec.image_folder))
    mirror = AbbaStackMirror(
        abba,
        paths,
        slice_by_id=slice_by_id,
        tasks=run_spec.tasks,
        sync_angles=run_spec.has("transform") and run_spec.transform.angles,
    )
    if run_spec.has("transform"):
        from PIL import Image

        from langslice.image_prep import read_pixel_size_um

        for path in paths:
            calibration = run_spec.inputs.get("pixel_size_um") or read_pixel_size_um(path)
            if (
                calibration is None
                or not math.isfinite(float(calibration))
                or float(calibration) <= 0
            ):
                raise ValueError("ABBA snapshots need a positive pixel-size calibration")
            with Image.open(path) as image:
                mirror._snapshot_geometry[os.path.basename(path)] = (
                    image.size,
                    float(calibration),
                )
    ordered = sorted(
        slice_by_id, key=lambda name: float(slice_by_id[name].getSlicingAxisPosition())
    )
    run_spec.inputs["order"] = ordered
    run_spec.inputs["positions"] = {
        name: mirror.from_abba_z(float(slice_by_id[name].getSlicingAxisPosition()))
        for name in ordered
    }
    run_spec.inputs["angles"] = {
        "pitch": math.degrees(float(resliced.getRotateX())) / PITCH_TO_ROTATE_X_SIGN,
        "yaw": math.degrees(float(resliced.getRotateY())) / YAW_TO_ROTATE_Y_SIGN,
    }
    metadata = {
        "slice_names": {name: str(sl.getName()) for name, sl in slice_by_id.items()},
        "axis_mapping": {"a": mirror._a, "b": mirror._b},
        "snapshot_pixel_size_um": run_spec.inputs.get("pixel_size_um"),
        "job": run_spec.to_dict(),
    }
    (Path(run_spec.image_folder) / "abba_run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    def observe(state: StackState) -> None:
        if mirror._last_state is None:
            # Ingestion describes the images just exported, not a request to
            # reset ABBA's existing orientation, registrations or positions.
            mirror._last_state = state.to_dict()
        else:
            mirror.on_write(state)
        if on_write is not None:
            try:
                on_write(state, slice_by_id)
            except Exception:
                logger.warning("ABBA progress display failed", exc_info=True)

    follower = None
    if follow_agent is not None:
        from langslice.integrations.abba_follow import AbbaFollower

        follower = AbbaFollower(
            abba, slice_by_id, enabled=follow_agent, comparison_factory=comparison_factory,
        )

    def events(event: dict[str, Any]) -> None:
        if follower is not None:
            follower.on_event(event)
        if on_event is not None:
            on_event(event)

    emit(f"[ABBA] Running on {len(ordered)} section(s); results in {run_spec.image_folder}")
    run_kwargs: dict[str, Any] = {"emit": emit, "on_write": observe}
    if on_event is not None or follower is not None:
        run_kwargs["on_event"] = events
    try:
        state = asyncio.run(run(run_spec, **run_kwargs))
        mirror.finish(state, save_state=save_state)
        return state
    finally:
        if follower is not None:
            follower.finish()



def run_linear_in_abba(
    spec: JobSpec,
    *,
    abba_atlas: str = "Adult Mouse Brain - Allen Brain Atlas V3p1",
    save_state: str | None = None,
    **nonlinear_overrides: Any,
) -> StackState:
    """Launch ABBA, import *spec*'s folder, and run the linear agent with a
    live mirror attached — the ``langslice abba --linear`` entry point.

    The agent is unchanged: it renders its own BrainGlobe pictures and
    drives itself exactly as it would headless. ABBA is display plus the
    final home of the result. The existing nonlinear registration plugin
    (``Register > LangSlice``) is installed in the same session —
    *nonlinear_overrides* forward to
    :func:`langslice.integrations.abba.enable_langslice_registration` — so
    both paths keep working side by side.
    """
    import asyncio

    import scyjava.config  # pyright: ignore[reportMissingImports]
    from abba_python.abba import Abba  # pyright: ignore[reportMissingImports]

    from langslice.integrations.abba import (
        GUI_DEPENDENCY,
        enable_langslice_registration,
        install_gui,
        wait_for_jvm_shutdown,
    )
    from langslice.linear.discovery import discover_slices
    from langslice.linear.engine import run

    scyjava.config.endpoints.append(GUI_DEPENDENCY)

    abba = Abba(abba_atlas)
    enable_langslice_registration(abba, **nonlinear_overrides)
    install_gui(abba)
    abba.show_bdv_ui()
    from langslice.integrations.abba_gui import MenuSettings, install_menu

    menu = install_menu(abba, settings=MenuSettings(
        model=spec.model or MenuSettings().model,
        reasoning=spec.reasoning or "medium",
        interval_um=spec.position.interval_um,
        thickness_um=spec.position.thickness_um,
        reorder=spec.has("reorder"), position=spec.has("position"),
        transform=spec.has("transform"), flip=spec.reorder.flip,
        hemisphere_cue=spec.reorder.hemisphere_cue,
        strict_interval=spec.position.strict_interval,
        interactive=spec.transform.interactive, automatic=spec.transform.automatic,
        elastix=spec.transform.elastix,
    ))

    folder = os.path.abspath(spec.image_folder)
    image_paths = discover_slices(folder)
    if not image_paths:
        raise ValueError(f"No slice images found in {folder}")

    with menu._lock:
        menu.running = True
    try:
        if menu.settings.open_agent_log and menu.activity_factory is not None:
            menu.activity_window = menu.activity_factory(spec.model or "")
        activity = menu.activity_window
        def emit(message: str) -> None:
            print(message)
            if activity is not None:
                activity.on_event({"kind": "progress", "text": message})
        mirror = AbbaStackMirror(abba, image_paths)
        from langslice.integrations.abba_follow import AbbaFollower

        follower = AbbaFollower(
            abba, mirror._slice_by_id, enabled=lambda: menu.settings.open_agent_viewer,
            comparison_factory=menu.comparison,
        )
        def events(event: dict[str, Any]) -> None:
            follower.on_event(event)
            if activity is not None:
                activity.on_event(event)
        try:
            state = asyncio.run(run(
                spec, emit=emit, on_write=mirror.on_write, on_event=events,
            ))
            mirror.finish(state, save_state=save_state)
        finally:
            follower.finish()
        if activity is not None:
            activity.set_status("Submitted" if state.submitted else "Stopped")
    except Exception as exc:
        if menu.activity_window is not None:
            menu.activity_window.on_event({"kind": "error", "text": str(exc)})
        raise
    finally:
        with menu._lock:
            menu.running = False

    wait_for_jvm_shutdown()
    return state
