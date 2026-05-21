"""Shared trace-handling primitives for SFT, iSFT, and single-turn RL."""

from ._empirical import sample, sample_roundness
from .accuracy import (
    canonicalize_positions,
    categorize_trace,
    plane_rescue_threshold_mm,
    plane_tolerance_mm,
)
from .augmentations import AtlasFetchJitter
from .constants import (
    DEFAULT_VLM_MAX_LONG_EDGE,
    DEFAULT_VLM_MAX_PIXELS,
    MODEL_POWER_TIER,
    RESCUE_PCT,
    TOLERANCE_FLOOR_MM,
    TOLERANCE_PCT,
)
from .generator import (
    CANONICAL_ATLAS_ROOT,
    GenerationStrategy,
    canonical_atlas_repo_path,
    generate_trace,
    load_atlas_grid,
)
from .image import (
    PreparedImage,
    adaptive_preprocess,
    normalize_image,
    prepare_image_for_vlm,
)
from .iterator import RenderMode, TraceIterator
from .manifest import TraceManifestRow, load_manifest, parse_manifest_record
from .parser import iter_canonical_traces, parse_canonical_trace
from .quality import classify_quality
from .reducer import best_traces
from .renderers import (
    render_isft_prefix,
    render_rl_prefix,
    render_sft_answer_only,
    render_sft_full,
)
from .schema import (
    CanonicalTrace,
    FinalAnswer,
    Plane,
    RenderedExample,
    RenderMetadata,
    ToolStep,
)
from .trace_ops import trim_trace

__all__ = [
    "adaptive_preprocess",
    "AtlasFetchJitter",
    "best_traces",
    "CANONICAL_ATLAS_ROOT",
    "canonical_atlas_repo_path",
    "canonicalize_positions",
    "CanonicalTrace",
    "categorize_trace",
    "classify_quality",
    "DEFAULT_VLM_MAX_LONG_EDGE",
    "DEFAULT_VLM_MAX_PIXELS",
    "FinalAnswer",
    "generate_trace",
    "GenerationStrategy",
    "iter_canonical_traces",
    "load_atlas_grid",
    "load_manifest",
    "MODEL_POWER_TIER",
    "normalize_image",
    "parse_canonical_trace",
    "parse_manifest_record",
    "Plane",
    "plane_rescue_threshold_mm",
    "plane_tolerance_mm",
    "prepare_image_for_vlm",
    "PreparedImage",
    "render_isft_prefix",
    "render_rl_prefix",
    "render_sft_answer_only",
    "render_sft_full",
    "RenderedExample",
    "RenderMetadata",
    "RenderMode",
    "RESCUE_PCT",
    "sample",
    "sample_roundness",
    "ToolStep",
    "TOLERANCE_FLOOR_MM",
    "TOLERANCE_PCT",
    "TraceIterator",
    "TraceManifestRow",
    "trim_trace",
]
