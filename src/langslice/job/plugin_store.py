"""What tool plugins keep in the job folder.

A plugin tool writes only under ``plugins/<tool>/`` (:meth:`JobLayout.plugin_dir`):
its own scratch, and the layer files it registers. The layers' files and the
manifest ``plugins/<tool>/layers.json`` (every layer the tool ever made, with
the call that made it) are here; which layers a section shows now is its
``layers`` in the state, so undoing the call that made a layer hides it and
redoing shows it again. The files stay on disk.

``exports/plugins/<tool>/`` holds a copy of the layers a section shows at the
last ``export_maps`` (:func:`export_layers`), with the manifest.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langslice.core.plugin_layers import save_layer
from langslice.core.plugins import Layer
from langslice.core.state import SliceState, StackState
from langslice.job.checkpoint import write_json_atomic
from langslice.job.layout import JobLayout

#: The manifest of a tool's layers, in its folder.
MANIFEST_FILE = "layers.json"
#: Under the tool's folder: the layer files.
LAYERS_SUBDIR = "layers"
#: Under ``exports/``: the copies of the layers.
EXPORT_SUBDIR = "plugins"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def manifest_path(layout: JobLayout, tool: str) -> Path:
    return layout.plugin_dir(tool) / MANIFEST_FILE


def read_manifest(layout: JobLayout, tool: str) -> dict[str, Any]:
    """The tool's manifest (``tool``, ``next``, ``layers``); empty when it has none."""
    try:
        held = json.loads(manifest_path(layout, tool).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tool": tool, "next": 1, "layers": []}
    return held if isinstance(held, dict) else {"tool": tool, "next": 1, "layers": []}


def register_layer(
    layout: JobLayout, record: SliceState, tool: str, call: str, layer: Layer, *,
    persist: bool = True,
) -> dict[str, Any]:
    """Write *layer*'s file under the tool's folder, list it in the manifest
    and return its reference (what the section's ``layers`` holds). A layer
    of the same name on the section replaces the earlier one's reference (the
    earlier file stays). *persist* False writes nothing (a dry run) and
    returns the reference the layer would have."""
    manifest = read_manifest(layout, tool)
    number = int(manifest.get("next", 1))
    name = layer.name or tool
    stem = (layout.plugin_dir(tool) / LAYERS_SUBDIR
            / f"{layout.section_name(record.id)}_{number:04d}_{_UNSAFE.sub('_', name)}")
    suffix = {"points": ".csv", "lines": ".json", "labels": ".png", "scalar": ".npz",
              "picture": ".png"}[layer.kind]
    if persist:
        written = save_layer(layer, stem)
        suffix = written.suffix
    ref = {
        "id": f"{tool}/{number}", "name": name, "kind": layer.kind, "frame": layer.frame,
        "file": layout.relative(stem.with_suffix(suffix)), "tool": tool, "call": call,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if persist:
        manifest["tool"] = tool
        manifest["next"] = number + 1
        manifest.setdefault("layers", []).append({**ref, "section": record.id})
        layout.plugin_dir(tool).mkdir(parents=True, exist_ok=True)
        write_json_atomic(str(manifest_path(layout, tool)), manifest)
    record.layers = [item for item in record.layers if item.get("name") != name] + [ref]
    return ref


def layer_rows(state: StackState) -> list[dict[str, Any]]:
    """Every layer the sections show, one row each (``section``, then the
    reference's fields), in stack order."""
    return [{"section": record.id, **ref}
            for record in state.in_order() for ref in record.layers]


def export_layers(layout: JobLayout, state: StackState) -> list[str]:
    """Copy the layers the sections show to ``exports/plugins/<tool>/`` with a
    manifest each; the paths written (relative to the job folder). Nothing
    when no section has a layer."""
    rows = layer_rows(state)
    written: list[str] = []
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tool.setdefault(str(row["tool"]), []).append(row)
    for tool, items in by_tool.items():
        target = layout.exports_dir / EXPORT_SUBDIR / tool
        target.mkdir(parents=True, exist_ok=True)
        listed = []
        for row in items:
            source = layout.resolve(row["file"])
            if not source.is_file():
                continue
            destination = target / source.name
            shutil.copy2(source, destination)
            written.append(layout.relative(destination))
            listed.append({**row, "file": source.name})
        write_json_atomic(str(target / MANIFEST_FILE), {"tool": tool, "layers": listed})
        written.append(layout.relative(target / MANIFEST_FILE))
    return written
