"""Shared constants for the langslice_traces package."""

from __future__ import annotations

# Position-estimation tolerance is plane-relative (a fraction of the atlas
# extent on the slice-normal axis). Without this, a 0.9mm rescue threshold is
# 7% of coronal AP (13.2mm Allen) but 16% of canonical sagittal hemisphere
# (5.7mm) — too loose for the shorter planes. These percentages give roughly
# coronal-equivalent strictness across all three planes.
TOLERANCE_PCT = 0.015  # accept threshold ≈ 0.20mm coronal Allen
RESCUE_PCT = 0.07      # rescue threshold ≈ 0.92mm coronal Allen
TOLERANCE_FLOOR_MM = 0.05


DEFAULT_VLM_MAX_PIXELS = 12_000_000
DEFAULT_VLM_MAX_LONG_EDGE = 4096


MODEL_POWER_TIER = {
    "gemini-3.1-pro-preview": "pro",
    "gemini-3-flash-preview": "flash3",
    "gemini-3.1-flash-lite-preview": "flash_lite",
    "gemini-2.5-flash": "flash2_5",
    "openrouter:qwen/qwen3.6-flash": "qwen_flash",
    "openrouter:qwen/qwen3.5-flash-02-23": "qwen_flash",
}
