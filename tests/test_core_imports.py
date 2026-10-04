"""The core library (and the job layer on it) imports without the agent
framework or any model client.

Each core module is imported in a fresh interpreter (a module another test
loaded would hide the leak) and must leave no ``google.adk``,
``google.genai``, ``litellm`` or ``openai`` module behind. Those belong to
the doors (``langslice.adk``, the toolbox, the MCP server), the agent driver
and the providers. The operations (``langslice.ops``) must also leave no
door module behind.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

CORE_MODULES = (
    "langslice.linear.workspace",
    "langslice.linear.render",
    "langslice.linear.display",
    "langslice.linear.transform",
    "langslice.linear.deformation",
    "langslice.linear.appearance",
    "langslice.linear.atlas_fetch",
    "langslice.linear.opening",
    "langslice.linear.job",
    "langslice.registration_handoff",
    "langslice.deformable",
    "langslice.space",
    "langslice.affine",
    "langslice.atlas",
    "langslice.core",
    "langslice.core.pictures",
    "langslice.core.placement",
    "langslice.ops",
    "langslice.ops.refusal",
    "langslice.ops.positions",
    "langslice.ops.order",
    "langslice.ops.orientation",
    "langslice.ops.damage",
    "langslice.ops.appearance",
    "langslice.ops.notes",
    "langslice.ops.transforms",
    "langslice.ops.deformable",
)

#: The doors: an operation (``langslice.ops``) must load none of them.
DOORS = (
    "langslice.linear.toolbox",
    "langslice.linear.view_options",
    "langslice.adk",
    "langslice.mcp_server",
)

FORBIDDEN = ("google.adk", "google.genai", "litellm", "openai")

_PROBE = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
forbidden = tuple(sys.argv[2:])
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)))
"""


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_loads_no_agent_or_model_client(module: str):
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *FORBIDDEN],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"{module} loads {loaded[:5]}"


@pytest.mark.parametrize("module", [name for name in CORE_MODULES
                                    if name.startswith("langslice.ops")])
def test_operations_load_no_door(module: str):
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *DOORS],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"{module} loads {loaded[:5]}"
