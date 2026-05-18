"""Stochastic augmentations applied to CanonicalTrace before rendering.

Each augmentation is a pure CanonicalTrace -> CanonicalTrace transform.
TraceIterator orchestrates application + seeded RNG.

The only augmentation in this pass is :class:`AtlasFetchJitter`. ClaheMixToggle
is deferred to Pass 4 because path-keying the SigLIP atlas embedding cache
makes per-call CLAHE toggling cache-invalidating; resolving that requires the
cache redesign and is out of scope here.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable, Optional

from .schema import CanonicalTrace, ToolStep


class AtlasFetchJitter:
    """Perturb fetch_atlas positions to nearby pre-embedded grid positions.

    For up to ``max_calls_jittered`` randomly-selected fetch_atlas steps,
    replace each position_mm with a perturbed value drawn from
    ``rng.gauss(orig, sigma_mm)``, then snap to the nearest pre-embedded
    atlas grid position. The snap step is critical: it keeps the trainer's
    path-keyed atlas embedding cache hitting (no live SigLIP fallback).

    The grid is supplied via ``grid_resolver(atlas_name, plane) -> list[float]``
    so the augmentation can be configured without depending on the cache
    object directly. If ``grid_resolver`` is ``None``, snapping is skipped
    (jittered values are used as-is). That mode is useful for tests but
    NOT recommended in training because cache misses force a live SigLIP
    forward pass on every fetch.
    """

    def __init__(
        self,
        *,
        sigma_mm: float,
        max_calls_jittered: int,
        grid_resolver: Optional[Callable[[str, str], list[float]]] = None,
    ):
        """Build a stateless jitter augmentation.

        Parameters
        ----------
        sigma_mm
            Standard deviation of the Gaussian perturbation (mm).
        max_calls_jittered
            Upper bound on the number of fetch_atlas steps to perturb per
            trace. If the trace has fewer tool_steps, all of them are
            jittered.
        grid_resolver
            Optional callable ``(atlas_name, plane) -> list[float]`` that
            returns the sorted list of pre-embedded mm positions for that
            (atlas, plane). When provided, every jittered position is
            snapped to its nearest grid entry so the path-keyed atlas
            embedding cache continues to hit.
        """
        self.sigma_mm = float(sigma_mm)
        self.max_calls_jittered = int(max_calls_jittered)
        self.grid_resolver = grid_resolver

    def __call__(self, canonical: CanonicalTrace, rng: random.Random) -> CanonicalTrace:
        """Apply jitter. Returns a NEW CanonicalTrace (input is not mutated)."""
        steps = canonical.tool_steps
        if not steps:
            return canonical

        # Pick which step indices to jitter. Cap at the actual number of
        # steps so we never index out of range; ``rng.sample`` requires
        # ``k <= len(population)``.
        n_to_jitter = min(self.max_calls_jittered, len(steps))
        if n_to_jitter <= 0:
            return canonical
        chosen = set(rng.sample(range(len(steps)), n_to_jitter))

        grid: list[float] | None = None
        if self.grid_resolver is not None:
            grid = list(self.grid_resolver(canonical.atlas_name, canonical.plane))

        new_steps: list[ToolStep] = []
        for i, step in enumerate(steps):
            if i not in chosen:
                new_steps.append(step)
                continue
            positions = step.call_args.get("positions_mm")
            if not isinstance(positions, list):
                # Nothing to jitter on this step — pass through unchanged.
                new_steps.append(step)
                continue
            jittered: list[float] = []
            for orig in positions:
                perturbed = float(orig) + rng.gauss(0.0, self.sigma_mm)
                if grid:
                    perturbed = _snap_to_grid(perturbed, grid)
                jittered.append(float(perturbed))
            new_call_args = dict(step.call_args)
            new_call_args["positions_mm"] = jittered
            new_steps.append(
                ToolStep(
                    call_name=step.call_name,
                    call_args=new_call_args,
                    result_image_paths=list(step.result_image_paths),
                    result_text=step.result_text,
                )
            )

        return replace(canonical, tool_steps=new_steps)


def _snap_to_grid(value: float, grid: list[float]) -> float:
    """Return the grid entry closest to value. grid must be non-empty."""
    if not grid:
        raise ValueError("grid must be non-empty")
    return min(grid, key=lambda g: abs(g - value))
