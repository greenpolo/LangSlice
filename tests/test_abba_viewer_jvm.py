"""The agent viewer against a real ABBA 0.24.1 JVM (headless).

Starts a JVM with ``abba_launch.ABBA_JAVA_DEPENDENCIES`` in a subprocess and
checks that every ABBA, BigDataViewer and bdv-playground member the viewer
(``abba_compare``, ``abba_overview``) and the launcher use exists, copies a
real source the way the viewer does, and, when the connector jar is built,
has the connector's ``LangSliceEvents`` deliver a message to the Python
``RunListener``. The fakes in
the other viewer tests cannot catch a renamed Java class.

Needs a Python with JPype and scyjava: this interpreter, or the one named by
``LANGSLICE_ABBA_PYTHON`` (e.g. the ``langslice`` conda env of the ABBA
installation). The first run downloads ABBA's jars into ``~/.jgo``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SRC = Path(__file__).resolve().parents[1] / "src"

SCRIPT = r'''
import json, sys
import scyjava, scyjava.config as cfg
from langslice.hosts.integrations import abba_compare, abba_launch, abba_overview

cfg.endpoints.extend(abba_launch.ABBA_JAVA_DEPENDENCIES)
try:
    jar = abba_launch.connector_jar()
    cfg.add_classpath(str(jar))
except FileNotFoundError:
    jar = None
cfg.add_option("-Djava.awt.headless=true")
cfg.set_java_constraints(fetch="auto", version=abba_launch.JAVA_VERSION)
scyjava.start_jvm()
from scyjava import jimport

Class = jimport("java.lang.Class")
used = {
    "ch.epfl.biop.atlas.aligner.MultiSlicePositioner":
        ["sX", "sY", "sizePixX", "getSlices", "getReslicedAtlas", "getAtlas"],
    "ch.epfl.biop.atlas.aligner.ReslicedAtlas":
        ["extendedSlicedSources", "nonExtendedSlicedSources", "getStep", "getRotateX",
         "getRotateY"],
    "ch.epfl.biop.atlas.aligner.SliceSources":
        ["getRegisteredSources", "getName", "getSlicingAxisPosition", "isKeySlice"],
    "ch.epfl.biop.atlas.aligner.gui.bdv.BdvMultislicePositionerView":
        ["getChannelVisibility", "getDisplaySettings", "getDisplayMode", "getDisplayedCenter",
         "getBdvh"],
    "ch.epfl.biop.atlas.aligner.gui.bdv.ABBABdvViewPrefs":
        ["dashed_stroke_slice_handle_to_atlas", "color_slice_handle_selected",
         "color_slice_handle_not_selected", "line_between_selected_slices_stroke",
         "line_between_selected_slices_color"],
    "ch.epfl.biop.atlas.aligner.gui.bdv.ABBATheme": ["setTheme", "createLightTheme"],
    "ch.epfl.biop.atlas.struct.AtlasMap": ["getImagesKeys", "getStructuralImages"],
    abba_overview.HANDLE_PACKAGE + ".CircleGraphicalHandle": ["enabledDraw"],
    abba_overview.HANDLE_PACKAGE + ".SquareGraphicalHandle": ["enabledDraw"],
    abba_overview.HANDLE_PACKAGE + ".GraphicalHandleListener": ["hover_in", "created"],
    "bdv.util.BdvHandlePanel": ["getViewerPanel", "getTriggerbindings", "close"],
    "bdv.util.BdvFunctions": ["show"],
    "bdv.util.BdvStackSource": ["removeFromBdv", "setActive"],
}
missing = []
for name, members in used.items():
    try:
        cls = Class.forName(name)
    except Exception:
        missing.append(name)
        continue
    have = {str(m.getName()) for m in cls.getMethods()} | {
        str(f.getName()) for f in cls.getFields()}
    missing += [f"{name}.{member}" for member in members if member not in have]

# The viewer's copy of a native source, on a real one.
img = jimport("net.imglib2.img.array.ArrayImgs").unsignedBytes(4, 4, 4)
source = jimport("bdv.util.RandomAccessibleIntervalSource")(
    img, jimport("net.imglib2.type.numeric.integer.UnsignedByteType")(), "probe")
native = jimport("sc.fiji.bdvpg.source.SourceHelper").createSourceAndConverter(source)
copied = abba_compare.copy_native_source(jimport, native)
copied.getSpimSource().setFixedTransform(jimport("net.imglib2.realtransform.AffineTransform3D")())

delivered = None
if jar is not None:
    listener = abba_launch.RunListener(slices=dict)
    received = []
    listener.handle = received.append
    forward = abba_launch._java_listener(listener)
    events = jimport(abba_launch.EVENTS_CLASS)
    events.addListener(forward)
    # The connector's own (package-private) publish, as a run calls it.
    String, JsonObject = jimport("java.lang.String"), jimport("com.google.gson.JsonObject")
    publish = Class.forName(abba_launch.EVENTS_CLASS).getDeclaredMethod(
        "publish", String.class_, JsonObject.class_)
    publish.setAccessible(True)
    body = JsonObject()
    body.addProperty("message", "probe")
    publish.invoke(None, "log", body)
    flush = Class.forName(abba_launch.EVENTS_CLASS).getDeclaredMethod("flush")
    flush.setAccessible(True)
    flush.invoke(None)
    listener.close()
    delivered = [json.loads(text) for text in received]
print(json.dumps({"missing": missing, "copied": str(copied.getSpimSource().getName()),
                  "delivered": delivered}))
'''


def _python() -> str | None:
    if all(importlib.util.find_spec(name) for name in ("jpype", "scyjava")):
        return sys.executable
    given = os.environ.get("LANGSLICE_ABBA_PYTHON", "").strip()
    return shutil.which(given) or (given if given and Path(given).is_file() else None)


def test_viewer_and_launcher_java_members_exist_in_abba_0_24_1():
    python = _python()
    if python is None:
        pytest.skip("No Python with JPype and scyjava (set LANGSLICE_ABBA_PYTHON)")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(SRC), os.environ.get("PYTHONPATH")])))
    done = subprocess.run([python, "-c", SCRIPT], capture_output=True, text=True, env=env,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    import json

    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["missing"] == []
    assert report["copied"] == "probe"
    if report["delivered"] is not None:
        assert report["delivered"] == [{"message": "probe", "kind": "log"}]
