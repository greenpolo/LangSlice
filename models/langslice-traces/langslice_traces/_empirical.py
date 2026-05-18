"""Empirical distributions of real Gemini-authored trace shape.

Frozen at extraction time 2026-05-10 from
models/langslice-gemma-4/data/sft_examples.jsonl (n=1716).

Regenerate by running models/langslice-traces/scripts/extract_empirical_distributions.py if the
SFT corpus is materially changed.

All corpus-extracted distributions are lists of (value, weight) tuples where
weights sum to 1.0. Use the sample() helper for reproducible draws via a
passed random.Random.

A small group of constants at the bottom of this module are Lane B-specific
scope choices rather than corpus extractions (the Lane B baseline has not yet
been measured -- these may be revisited once it has). They live here so all
generator knobs are colocated.
"""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")

# Number of tool steps per trace (after dropping the terminal submit step).
P_N_TOOL_STEPS: list[tuple[int, float]] = [
    (1, 0.009907),
    (2, 0.990093),
]

# Number of positions per fetch, step 0 (broad).
P_NFETCH_STEP0: list[tuple[int, float]] = [
    (1, 0.002914),
    (2, 0.014569),
    (3, 0.076923),
    (4, 0.296037),
    (5, 0.482517),
    (6, 0.089744),
    (7, 0.026224),
    (8, 0.011072),
]

# Number of positions per fetch, step 1 (narrow).
P_NFETCH_STEP1: list[tuple[int, float]] = [
    (1, 0.071807),
    (2, 0.131842),
    (3, 0.297234),
    (4, 0.180106),
    (5, 0.257799),
    (6, 0.035315),
    (7, 0.023543),
    (8, 0.002354),
]

# Span (max - min) per step, in mm. Bins of (span_mm, weight).
P_SPAN_STEP0: list[tuple[float, float]] = [
    (1.0000, 0.513403),
    (3.0000, 0.131119),
    (5.0000, 0.234848),
    (7.0000, 0.056527),
    (9.0000, 0.049534),
    (11.0000, 0.002331),
    (13.0000, 0.009324),
    (15.0000, 0.001748),
    (17.0000, 0.000583),
    (19.0000, 0.000583),
]

P_SPAN_STEP1: list[tuple[float, float]] = [
    (0.4000, 0.801059),
    (1.2000, 0.167157),
    (2.0000, 0.021778),
    (2.8000, 0.001766),
    (3.6000, 0.000589),
    (4.4000, 0.005886),
    (5.2000, 0.000589),
    (7.6000, 0.001177),
]

# Signed offset between step-1 center and submit position (mm).
P_CENTER_OFFSET_STEP1: list[tuple[float, float]] = [
    (-1.9000, 0.000589),
    (-0.9000, 0.001177),
    (-0.7000, 0.000589),
    (-0.5000, 0.005886),
    (-0.3000, 0.025898),
    (-0.1000, 0.284873),
    (0.1000, 0.642142),
    (0.3000, 0.030606),
    (0.5000, 0.006474),
    (1.1000, 0.000589),
    (1.9000, 0.001177),
]

# Roundness class probabilities per step.
P_ROUNDNESS_STEP0: dict[str, float] = {"int": 0.495134, "half": 0.119042, "tenth": 0.347766, "raw": 0.038058}
P_ROUNDNESS_STEP1: dict[str, float] = {"int": 0.109869, "half": 0.119592, "tenth": 0.521147, "raw": 0.249392}

# Sigma for the step-0 anchor in the slope-1 model: sampling
# ``center_0 = gt + N(0, SIGMA_ANCHOR_MM)`` reproduces the observed
# distribution of step-0 fetch centers around GT (population std of
# center_0 - gt; observed Pearson r ~ 0.982 on the corpus).
SIGMA_ANCHOR_MM: float = 0.891648

# Per-plane breakdown of the same sigma (planes have different extents).
SIGMA_ANCHOR_MM_BY_PLANE: dict[str, float] = {"coronal": 0.891980, "horizontal": 0.955550, "sagittal": 0.820223}

# Diagnostic - what the source corpus actually shows.
OBSERVED_R_STEP0_GT: float = 0.981682
OBSERVED_R_STEP0_GT_BY_PLANE: dict[str, float] = {"coronal": 0.983136, "horizontal": 0.894835, "sagittal": 0.888796}
OBSERVED_INTEGER_RATE_STEP0: float = 0.495134
OBSERVED_N: int = 1716


# ---------------------------------------------------------------------------
# Lane B-specific constants (not extracted from corpus -- scope choices that
# we may revisit when we measure Lane B baselines).
# ---------------------------------------------------------------------------

# Slate width clamp range. The broad-step empirical distribution
# (:data:`P_SPAN_STEP0`) skews narrow (>50% at 1mm); a slate that narrow is
# not "broad" enough to be Lane-B-shaped. We clip into a band that always
# covers ~10% of a coronal extent.
LANE_B_WIDTH_LO_MM: float = 3.0
LANE_B_WIDTH_HI_MM: float = 12.0

# When clipping the slate to plane bounds would push GT to within this
# fraction of an edge of the slate, redraw the fraction once. Caps a
# pathological geometry but doesn't loop -- a second-roll failure means the
# plane is just too small.
LANE_B_EDGE_GUARD: float = 0.05

# Lane B n_positions distribution (uniform over 7, 8, 9).
P_NFETCH_LANE_B: list[tuple[int, float]] = [
    (7, 1.0 / 3.0),
    (8, 1.0 / 3.0),
    (9, 1.0 / 3.0),
]


def sample(rng: random.Random, dist: list[tuple[T, float]]) -> T:
    """Draw one value from a weighted (value, prob) distribution.

    Uses ``rng.random()`` for reproducibility; weights must sum to ~1.0 but
    we use cumulative comparison so non-normalized inputs still work.
    """
    total = sum(w for _, w in dist)
    target = rng.random() * total
    cumulative = 0.0
    for value, weight in dist:
        cumulative += weight
        if cumulative >= target:
            return value
    return dist[-1][0]  # numerical-precision fallback


def sample_roundness(rng: random.Random, dist: dict[str, float]) -> str:
    """Draw a roundness class ('int'|'half'|'tenth'|'raw') from a dict dist."""
    return sample(rng, [(k, v) for k, v in dist.items()])
