"""LangSlice: register histology sections to BrainGlobe atlases.

Start here. The library a script uses, one import path, nothing of the agent
framework loaded::

    import langslice

    job = langslice.open_job("/data/brain1")   # a job folder, or its images
    help(job)                                   # every tool of this job
    job.status()                                # a tool, as a method
    reply = job.look(mode="overlay", sections=["s01.tif"])
    reply["artifacts"]                          # the picture files

Where everything is:

- ``help(job)``: the job's tools, grouped, each a method with the agent
  tool's name and arguments (``help(job.look)``), and the handle's own
  methods: ``wait``, ``close``, ``section_maps``, ``export_stack``,
  ``export_abba``, ``export_dmc_brainmap``.
- ``help(job.context)``: a method of your own written into the job (your own
  affine, warp field or section image, a preview that writes nothing), in six
  parts, each with its own ``help``: ``reads`` (sections, atlas, images,
  masks), ``frames`` (fit grid, atlas planes, points to atlas micrometres),
  ``proposals`` (a transform or deformation as LangSlice stores it),
  ``fits`` (FireANTs, elastix, an outside engine's ITK files), ``writes``
  and ``pictures``.
- :func:`open_job`, :func:`create_job` (:mod:`langslice.doors.library.handle`):
  open a job, or make one from a folder of sections.
- :func:`coordinate_map` (:mod:`langslice.core.pictures.layers`): a saved
  picture's pixels in atlas micrometres, ``(rows, cols, 3)`` float32, the
  atlas's own axis order, voxel ``i``'s centre at ``i * resolution``.
- :func:`load_atlas` (:mod:`langslice.core.atlas.load`): a cached BrainGlobe
  atlas.
- :func:`image_model`, :func:`default_prompt`
  (:mod:`langslice.providers.profiles`): the image model ``trace_borders``
  calls (a provider's model and LangSlice's prompt, or your own), and
  LangSlice's prompt text to start from.
- :mod:`langslice.plugin_api`: the types the context parts return, and how to
  add a tool of your own to a job.
- The job folder's ``AGENTS.md``: its files (``state.json`` is the truth,
  the rest is derived) and its coordinates.

Each name is loaded on first use, so ``import langslice`` stays light.
``docs/developers/library.md`` is the scripting guide.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any

__version__ = "0.2.1"
__all__ = [
    "__version__",
    "coordinate_map",
    "create_job",
    "default_prompt",
    "image_model",
    "load_atlas",
    "open_job",
]

if TYPE_CHECKING:
    from langslice.core.atlas.load import load_atlas as load_atlas
    from langslice.core.pictures.layers import coordinate_map as coordinate_map
    from langslice.doors.library.handle import create_job as create_job
    from langslice.doors.library.handle import open_job as open_job
    from langslice.providers.profiles import default_prompt as default_prompt
    from langslice.providers.profiles import image_model as image_model

_PUBLIC = {
    "open_job": ("langslice.doors.library.handle", "open_job"),
    "create_job": ("langslice.doors.library.handle", "create_job"),
    "image_model": ("langslice.providers.profiles", "image_model"),
    "default_prompt": ("langslice.providers.profiles", "default_prompt"),
    "coordinate_map": ("langslice.core.pictures.layers", "coordinate_map"),
    "load_atlas": ("langslice.core.atlas.load", "load_atlas"),
}


def __getattr__(name: str) -> Any:
    if name in _PUBLIC:
        import importlib

        module, attribute = _PUBLIC[name]
        value = getattr(importlib.import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'langslice' has no attribute {name!r}")


class _GeminiNonTextPartsFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


_logging.getLogger("google_genai.types").addFilter(_GeminiNonTextPartsFilter())
