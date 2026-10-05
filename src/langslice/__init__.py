"""LangSlice: register histology sections to BrainGlobe atlases.

The library a script uses (one import path, nothing of the agent framework
loaded)::

    import langslice

    job = langslice.open_job("/data/brain1")   # a job folder, or its images
    job.status()                                # the verbs, as methods
    xyz = langslice.coordinate_map(".../view.json")   # atlas µm per pixel
    atlas = langslice.load_atlas("allen_mouse_25um")  # a BrainGlobe atlas

    # A scripted pipeline: the image model's nonlinear registration, no agent.
    model = langslice.image_model("openai-oauth")     # or a profile of your own
    result = langslice.register_section("s01.tif", position_mm=6.2, image_model=model)
    job = langslice.create_job("/data/brain1", positions=..., image_model=model)
    result = langslice.register_job(job)

- :func:`open_job`, :func:`create_job` (:mod:`langslice.doors.library`): the
  job's verbs as methods, the same names and arguments as the agent tools
  and ``langslice job FOLDER VERB`` (``langslice ops``, ``langslice schema
  VERB``); ``create_job`` makes the job of a folder of sections.
- :func:`image_model`, :func:`default_prompt`
  (:mod:`langslice.providers.profiles`): the image model profile the border
  trace calls (a provider's model and LangSlice's prompt, or a model and
  prompt of your own), and LangSlice's prompt text to start from.
- :func:`register_section`, :func:`register_job`
  (:mod:`langslice.doors.pipeline`): the scripted registration (the
  automatic linear alignment where needed, the image model's trace, the
  deformable fit, the maps and exports) on one section or a whole job;
  :class:`RegistrationError` when a section could not be registered.
- :func:`coordinate_map` (:mod:`langslice.core.layers`): a saved picture's
  pixels in atlas micrometres, ``(rows, cols, 3)`` float32, the atlas's own
  axis order, voxel ``i``'s centre at ``i * resolution``.
- :func:`load_atlas` (:mod:`langslice.core.atlas.core`): a cached BrainGlobe atlas.

Each is loaded on first use, so ``import langslice`` stays light.
``docs/library.md`` is the pipeline-integration guide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = [
    "RegistrationError", "__version__", "coordinate_map", "create_job", "default_prompt",
    "image_model", "load_atlas", "open_job", "register_job", "register_section",
]

if TYPE_CHECKING:
    from langslice.core.atlas.core import load_atlas as load_atlas
    from langslice.core.layers import coordinate_map as coordinate_map
    from langslice.doors.library import create_job as create_job
    from langslice.doors.library import open_job as open_job
    from langslice.doors.pipeline import RegistrationError as RegistrationError
    from langslice.doors.pipeline import register_job as register_job
    from langslice.doors.pipeline import register_section as register_section
    from langslice.providers.profiles import default_prompt as default_prompt
    from langslice.providers.profiles import image_model as image_model

#: The public names, each loaded from its module on first use.
_PUBLIC = {
    "open_job": ("langslice.doors.library", "open_job"),
    "create_job": ("langslice.doors.library", "create_job"),
    "image_model": ("langslice.providers.profiles", "image_model"),
    "default_prompt": ("langslice.providers.profiles", "default_prompt"),
    "register_section": ("langslice.doors.pipeline", "register_section"),
    "register_job": ("langslice.doors.pipeline", "register_job"),
    "RegistrationError": ("langslice.doors.pipeline", "RegistrationError"),
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


# A response whose only part is a function call makes google-genai's
# response.text log a WARNING ("non-text parts in the response") through the
# `google_genai.types` logger. The agent loop reads function calls directly
# (`event.get_function_calls()`), so the warning is noise that reads like an
# error in the run's log. Silence that one message; other google_genai
# warnings stay visible.
import logging as _logging  # noqa: E402


class _GeminiNonTextPartsFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


_logging.getLogger("google_genai.types").addFilter(_GeminiNonTextPartsFilter())
