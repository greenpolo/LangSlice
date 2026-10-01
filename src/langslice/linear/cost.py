"""Pre-run cost estimate for a linear stack run on the ChatGPT account.

The unit is percent of the ChatGPT usage window, the only cost the OAuth lane
reports. Rates are percent per section, measured from 92 past runs whose logs
carry usage-window readings (2026-09-09 to 2026-09-23). Every one of those runs
had 36 or 38 sections and used the picture sizes of that period (every
picture at the atlas's 25 um, roughly 200-450 px), so the estimate is a linear
extrapolation in stack size and is offered only at "low". Since 2026-10-01
"low" opens at 256 px and draws later pictures at 512 px
(``render.PICTURE_EDGES``), larger than those runs saw: no run has been
measured at the new sizes yet. Medium, high and auto are refused. The window
reading is shared by the whole account and has one-percent resolution, so every
figure here is an estimate, not a meter.
"""

from __future__ import annotations

from typing import Any

#: (model, reasoning, transform task on) -> (low, high) percent per section and
#: the number of measured runs. Several runs: the 20th-80th percentile band.
#: One run: that single value, widened below.
_RATES: dict[tuple[str, str, bool], tuple[float, float, int]] = {
    ("gpt-6-astra", "low", False): (0.450, 0.633, 2),
    ("gpt-6-astra", "low", True): (0.744, 0.894, 2),
    ("gpt-6-astra", "medium", False): (0.778, 0.778, 1),
    ("gpt-6-astra", "medium", True): (0.132, 0.447, 2),
    ("gpt-5.6-sol", "medium", False): (0.167, 0.167, 1),
    ("gpt-5.6-sol", "medium", True): (0.222, 0.553, 23),
    # Luna at low reasoning rounds to zero on the one-percent reading; the
    # upper value is the resolution floor, not a measurement.
    ("gpt-5.6-luna", "low", False): (0.0, 0.011, 4),
    ("gpt-5.6-luna", "high", False): (0.028, 0.028, 1),
    ("gpt-5.6-luna", "high", True): (0.026, 0.105, 43),
    ("gpt-6-luna", "medium", True): (0.083, 0.083, 1),
    ("gpt-6-luna", "xhigh", True): (0.028, 0.028, 1),
    ("gpt-6-sol", "medium", True): (0.080, 0.082, 2),
}
#: A single run, or a neighbouring setting, is widened by this factor each way.
_WIDEN = 2.0
_REASONING_ORDER = ("low", "medium", "high", "xhigh", "max")


def _model_name(model: str) -> str:
    return model.split("/", 1)[-1].strip()


def _rate(model: str, reasoning: str, transform: bool) -> tuple[float, float, str]:
    """(low, high, basis) percent per section for one setting."""
    exact = _RATES.get((model, reasoning, transform))
    if exact is not None:
        low, high, runs = exact
        if runs >= 2:
            return low, high, f"{runs} measured runs"
        return low / _WIDEN, high * _WIDEN, "1 measured run, widened"
    same_model = [
        (key, value) for key, value in _RATES.items() if key[0] == model
    ]
    if same_model:
        def distance(key: tuple[str, str, bool]) -> tuple[int, int]:
            wanted = _REASONING_ORDER.index(reasoning)
            return (abs(_REASONING_ORDER.index(key[1]) - wanted), int(key[2] != transform))

        key, (low, high, _runs) = min(same_model, key=lambda item: distance(item[0]))
        return (low / _WIDEN, high * _WIDEN,
                f"nearest measured setting ({key[1]} reasoning), widened")
    lows = [value[0] for value in _RATES.values()]
    highs = [value[1] for value in _RATES.values()]
    return min(lows), max(highs), "no runs with this model; range over all models"


def estimate(spec: dict[str, Any], n_slices: int, locked: int = 0) -> dict[str, Any]:
    """Estimated usage-window share for *n_slices* sections under *spec*.

    *spec* uses JobSpec field names. *locked* sections need no transform
    work; when positioning is off they need no work at all. Raises
    ``ValueError`` where no estimate can honestly be given.
    """
    from langslice.linear.spec import DEFAULT_MAX_QUOTA_PERCENT, JobSpec

    if n_slices < 1:
        raise ValueError("n_slices must be at least 1")
    if not 0 <= locked <= n_slices:
        raise ValueError("locked must be between 0 and n_slices")
    parsed = JobSpec.from_dict({**spec, "image_folder": spec.get("image_folder") or "."})
    if parsed.image_resolution != "low":
        raise ValueError("No runs have been measured at this image resolution")
    tasks = set(parsed.tasks)
    positioning = bool(tasks & {"reorder", "position"})
    transform = "transform" in tasks
    if not (positioning or transform):
        raise ValueError("No agent task is switched on")
    model = _model_name(parsed.model or "")
    if not model:
        from langslice.providers.openai_oauth import DEFAULT_REVIEW_MODEL

        model = _model_name(DEFAULT_REVIEW_MODEL)
    # The provider's own default effort; the measured runs that left it
    # unset ran at medium.
    reasoning = parsed.reasoning or "medium"
    if reasoning not in _REASONING_ORDER:
        raise ValueError(f"Unknown reasoning level {reasoning!r}")

    low_rate, high_rate, basis = _rate(model, reasoning, transform)
    # Every measured run positioned the whole stack, so a transform-only run
    # is priced as a full run over the sections it aligns: an upper bound.
    sections = n_slices if positioning else n_slices - locked
    if sections == 0:
        return {"low": 0.0, "high": 0.0, "unit": "percent_of_usage_window",
                "basis": "every section is locked"}
    low, high = low_rate * sections, high_rate * sections
    cap = float(spec.get("max_quota_percent", DEFAULT_MAX_QUOTA_PERCENT))
    if high > cap:
        high = cap
        basis += f"; a run stops at its {cap:g}% cap"
    low = min(low, high)
    if n_slices < 20 or n_slices > 60:
        basis += "; measured stacks had 36-38 sections"
    return {"low": round(low, 1), "high": round(high, 1),
            "unit": "percent_of_usage_window", "basis": basis}
