"""LangSlice: register histology sections to BrainGlobe atlases.

The library a script uses (one import path, nothing of the agent framework
loaded)::

    import langslice

    job = langslice.open_job("/data/brain1")   # a job folder, or its images
    job.status()                                # the verbs, as methods
    xyz = langslice.coordinate_map(".../view.json")   # atlas µm per pixel
    atlas = langslice.load_atlas("allen_mouse_25um")  # a BrainGlobe atlas

    model = langslice.image_model("openai-oauth")     # or a profile of your own
    job = langslice.create_job("/data/brain1", positions=..., image_model=model)
    job.trace_borders(section="s01.tif")              # any verb, by name

- :func:`open_job`, :func:`create_job` (:mod:`langslice.doors.library`): the
  job's verbs as methods, the same names and arguments as the agent tools
  and ``langslice-job FOLDER VERB`` (``langslice-job ops``, ``langslice-job schema
  VERB``); ``create_job`` makes the job of a folder of sections.
- :func:`image_model`, :func:`default_prompt`
  (:mod:`langslice.providers.profiles`): the image model profile the border
  trace calls (a provider's model and LangSlice's prompt, or a model and
  prompt of your own), and LangSlice's prompt text to start from.
- :func:`coordinate_map` (:mod:`langslice.core.layers`): a saved picture's
  pixels in atlas micrometres, ``(rows, cols, 3)`` float32, the atlas's own
  axis order, voxel ``i``'s centre at ``i * resolution``.
- :func:`load_atlas` (:mod:`langslice.core.atlas.core`): a cached BrainGlobe atlas.

Each is loaded on first use, so ``import langslice`` stays light.
``docs/library.md`` is the scripting guide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = [
    "__version__", "coordinate_map", "create_job", "default_prompt", "image_model",
    "load_atlas", "open_job",
]

if TYPE_CHECKING:
    from langslice.core.atlas.core import load_atlas as load_atlas
    from langslice.core.layers import coordinate_map as coordinate_map
    from langslice.doors.library import create_job as create_job
    from langslice.doors.library import open_job as open_job
    from langslice.providers.profiles import default_prompt as default_prompt
    from langslice.providers.profiles import image_model as image_model

#: The public names, each loaded from its module on first use.
_PUBLIC = {
    "open_job": ("langslice.doors.library", "open_job"),
    "create_job": ("langslice.doors.library", "create_job"),
    "image_model": ("langslice.providers.profiles", "image_model"),
    "default_prompt": ("langslice.providers.profiles", "default_prompt"),
    "coordinate_map": ("langslice.core.layers", "coordinate_map"),
    "load_atlas": ("langslice.core.atlas.core", "load_atlas"),
}


def __getattr__(name: str) -> Any:
    if name in _PUBLIC:
        import importlib

        module, attribute = _PUBLIC[name]
        value = getattr(importlib.import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'langslice' has no attribute {name!r}")


# google-genai warns through the `google_genai.types` logger whenever
# `response.text` is read from a response whose only part is a function call.
# The agent loop reads function calls with `event.get_function_calls()`, so
# the warning is noise that reaches every host's log as if it were an error.
# Only that message is silenced; other google_genai warnings stay visible.
import logging as _logging  # noqa: E402


class _GeminiNonTextPartsFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


_logging.getLogger("google_genai.types").addFilter(_GeminiNonTextPartsFilter())
