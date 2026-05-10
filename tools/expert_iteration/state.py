"""Per-run state tracking for multi-round expert iteration.

The orchestrator writes a ``state.json`` after every completed phase boundary
so a crashed run can resume without re-doing expensive work (rollouts, SFT
retrain, slicebench eval). Resumability is **phase-level**, not step-level —
granular enough to skip the rollouts on a crash mid-train, simple enough that
the resume logic is a single switch statement.

Phase ordering for round k:

    sampled    → prompts have been sampled and persisted to specs.jsonl
    rollouts   → all rollouts have been collected into round_<k>.jsonl
    scored     → scored predictions are on disk (round_<k>_scored.json)
    filtered   → filter has been applied, kept rows persisted
    appended   → kept rows appended to the iterative-corpus round JSONL
    unioned    → unioned + path-rewritten corpus is on disk
    trained    → SFT retrain finished, round_<k>_adapter exists
    evaluated  → slicebench eval finished, summary.json saved
    done       → round k complete; advance to k+1

When the driver starts and finds state.json, it reads ``round`` and ``phase``
to determine where to pick up. Phases are checked via ``state.is_phase_done``;
re-running a phase is idempotent because every phase writes deterministic
file artifacts the next phase consumes.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Ordered list of phases per round. ``done`` is the terminal sentinel meaning
# the round itself is complete and the driver should advance to round k+1.
PHASES: tuple[str, ...] = (
    "sampled",
    "rollouts",
    "scored",
    "filtered",
    "appended",
    "unioned",
    "trained",
    "evaluated",
    "done",
)


def _phase_index(phase: str) -> int:
    try:
        return PHASES.index(phase)
    except ValueError as exc:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}") from exc


@dataclass
class RunState:
    """Mutable per-run state.

    ``round`` is the index of the round currently in progress. ``phase``
    is the **last completed** phase within that round. A fresh run with
    no state.json starts at ``round=0, phase=None`` (which is treated
    as "nothing done yet for round 0").
    """

    run_id: str
    rounds_total: int
    round: int = 0
    phase: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    # ``history`` accumulates per-round summaries so a crashed mid-multi-round
    # job's diagnostics aren't lost when state.json gets overwritten.
    history: list[dict[str, Any]] = field(default_factory=list)

    # ────────────────────────────────────────────────────────────────────
    # Phase predicates
    # ────────────────────────────────────────────────────────────────────

    def is_phase_done(self, round_idx: int, phase: str) -> bool:
        """Returns True if the given phase of the given round is already done."""
        if round_idx < self.round:
            return True  # any earlier round is fully complete
        if round_idx > self.round:
            return False
        if self.phase is None:
            return False
        return _phase_index(self.phase) >= _phase_index(phase)

    def next_phase(self, round_idx: int) -> str:
        """Return the first phase that still needs to run for this round."""
        if round_idx < self.round:
            return "done"
        if round_idx > self.round or self.phase is None:
            return PHASES[0]
        idx = _phase_index(self.phase)
        if idx + 1 >= len(PHASES):
            return "done"
        return PHASES[idx + 1]

    # ────────────────────────────────────────────────────────────────────
    # Mutators
    # ────────────────────────────────────────────────────────────────────

    def mark_phase(self, round_idx: int, phase: str, **artifacts: Any) -> None:
        """Mark ``phase`` as the last completed step for ``round_idx``."""
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
        if round_idx != self.round:
            # Cross-round transition: previous round must have ended at "done"
            # (or this is a forward jump from a fresh state).
            if round_idx > self.round and self.phase != "done" and self.phase is not None:
                logger.warning(
                    "Advancing from round %d (phase=%s) to round %d without "
                    "completing the prior round.",
                    self.round, self.phase, round_idx,
                )
            self.round = round_idx
            self.phase = None
        self.phase = phase
        if artifacts:
            self.artifacts.setdefault(str(round_idx), {}).update(artifacts)

    def advance_round(self, summary: dict[str, Any] | None = None) -> None:
        """Move to the next round; archive the current round's artifacts/summary."""
        snapshot = {
            "round": self.round,
            "artifacts": dict(self.artifacts.get(str(self.round), {})),
        }
        if summary:
            snapshot["summary"] = summary
        self.history.append(snapshot)
        self.round += 1
        self.phase = None

    # ────────────────────────────────────────────────────────────────────
    # Serialization
    # ────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunState":
        return cls(
            run_id=str(payload.get("run_id", "")),
            rounds_total=int(payload.get("rounds_total", 1)),
            round=int(payload.get("round", 0)),
            phase=payload.get("phase"),
            artifacts=dict(payload.get("artifacts", {})),
            history=list(payload.get("history", [])),
        )


# ──────────────────────────────────────────────────────────────────────────
# Disk I/O
# ──────────────────────────────────────────────────────────────────────────

def state_path(output_dir: Path) -> Path:
    return Path(output_dir) / "state.json"


def load_state(output_dir: Path) -> RunState | None:
    """Load state.json from output_dir, returning None if absent or unparseable."""
    p = state_path(output_dir)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse %s (%s); treating as fresh run.", p, exc)
        return None
    return RunState.from_dict(payload)


def save_state(state: RunState, output_dir: Path) -> Path:
    """Atomically persist state to output_dir/state.json. Returns the path written."""
    out = state_path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True)
    # Atomic write: tempfile in the same dir, then os.replace.
    fd, tmp = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=out.parent)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
        Path(tmp).replace(out)
    except Exception:
        try:
            Path(tmp).unlink()
        except FileNotFoundError:
            pass
        raise
    return out


def init_or_resume(output_dir: Path, *, run_id: str, rounds_total: int) -> RunState:
    """Load existing state.json or initialize a fresh one (and persist it).

    If an existing state.json's ``run_id`` differs from the requested one,
    we keep the existing state (the directory is the source of truth) but
    log a warning. ``rounds_total`` is updated in either case so a resume
    with extra rounds (--rounds 5 → --rounds 8) extends the run.
    """
    existing = load_state(output_dir)
    if existing is not None:
        if existing.run_id and run_id and existing.run_id != run_id:
            logger.warning(
                "state.json run_id=%r differs from requested run_id=%r; "
                "honoring on-disk state.", existing.run_id, run_id,
            )
        existing.rounds_total = rounds_total
        save_state(existing, output_dir)
        return existing
    fresh = RunState(run_id=run_id, rounds_total=rounds_total)
    save_state(fresh, output_dir)
    return fresh
