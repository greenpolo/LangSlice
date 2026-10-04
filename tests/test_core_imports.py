"""The core library imports without the agent framework or any model client.

Each core module is imported in a fresh interpreter (a module another test
loaded would hide the leak) and must leave no ``google.adk``,
``google.genai``, ``litellm`` or ``openai`` module behind. Those belong to
the doors (``langslice.adk``, the toolbox, the MCP server), the agent driver
and the providers.
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
    "langslice.registration_handoff",
    "langslice.deformable",
    "langslice.space",
    "langslice.affine",
    "langslice.atlas",
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
