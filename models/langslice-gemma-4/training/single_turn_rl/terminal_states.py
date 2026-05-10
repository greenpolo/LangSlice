"""Terminal-state extraction for single-turn GRPO (Lane A).

A "terminal state" is the prompt the policy should see at the moment it would
otherwise call ``submit_estimate`` — the query image plus every atlas
reference the multi-turn agent had already fetched. Single-turn GRPO trains
the final coordinate decision against that frozen visual context.

This module owns:

1. :class:`TerminalState` — the on-disk row shape.
2. :func:`read_terminal_states` / :func:`write_terminal_states` — JSONL I/O.
3. :class:`ManifestGTLookup` — joins SFT rows to true ``position_mm`` from
   ``data/manifest/shards/<plane>/<dataset>.jsonl``.
4. :func:`build_from_sft_corpus` — walker that turns each accepted SFT trace
   into one terminal-state row by cutting the trace right before the
   ``submit_estimate`` step. The walker:

   * parses positions from atlas filenames (``<X.XX>mm.jpg``) so the
     captions in training match the actual snapped slice positions
     returned by the production tool — never the (potentially deduped /
     snapped) positions from ``tool_call.args``;
   * resolves true ground truth from the manifest shard, dropping rows
     that don't join (no silent fallback to the teacher's submit, which
     would cap policy accuracy at the teacher's accuracy);
   * stashes the teacher's submit position in
     ``quality["teacher_position_mm"]`` for diagnostic comparison only.

CLI
---
::

    python -m single_turn_rl.terminal_states build \\
        --sft-corpus models/langslice-gemma-4/data/sft_examples.jsonl \\
        --output out/single_turn_rl/terminal_states.jsonl \\
        --tier strict \\
        --manifest-root data/manifest

Run from the repo root with ``PYTHONPATH=models/langslice-gemma-4/training``.
``--repo-root`` defaults to the current working directory so the emitted
JSONL paths land in the form ``models/langslice-gemma-4/data/queries/...``,
matching the trainer's ``--repo-root .`` default.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Plane = Literal["coronal", "sagittal", "horizontal"]
TierFilter = Literal["strict", "rescued", "any"]

_VALID_TIERS: tuple[TierFilter, ...] = ("strict", "rescued", "any")

# Atlas reference images are named ``<X.XX>mm.jpg`` (or .png/.jpeg). The
# regex captures the numeric prefix so the walker can derive snapped
# positions directly from the on-disk filenames the production tool
# returned. Using filename-derived positions side-steps the snap/dedupe
# mismatch that ``tool_call.args.positions_mm`` can introduce when the
# production tool collapses near-duplicate fetches before rendering.
_ATLAS_POSITION_RE = re.compile(r"(\d+(?:\.\d+)?)mm\.(?:jpg|png|jpeg)$", re.IGNORECASE)


def _position_mm_from_atlas_path(path: str) -> float | None:
    """Parse ``<X.XX>mm`` out of an atlas filename. Returns None if unparseable."""
    m = _ATLAS_POSITION_RE.search(str(path).replace("\\", "/"))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


@dataclass(frozen=True)
class TerminalState:
    """One row of the terminal-state JSONL.

    Image paths are stored as repo-relative strings (resolved at decode time)
    so the JSONL is portable across machines that share the same checkout.

    ``ground_truth_mm`` is the manifest's ``position_mm`` for this section —
    NOT the SFT teacher's submit. The teacher's submit is preserved under
    ``quality["teacher_position_mm"]`` so callers can diagnose
    teacher-vs-truth divergence without using it as a training signal.
    """

    section_id: str
    subject_id: str
    atlas_name: str
    plane: Plane
    valid_range_mm: tuple[float, float]
    ground_truth_mm: float
    query_image_path: str
    atlas_image_paths: tuple[str, ...]
    fetched_positions_mm: tuple[float, ...]
    source: str  # e.g. "sft_corpus:strict"
    quality: dict[str, Any]  # passthrough provenance from the SFT row + teacher_position_mm

    def to_jsonl(self) -> str:
        payload = asdict(self)
        payload["valid_range_mm"] = list(self.valid_range_mm)
        payload["atlas_image_paths"] = list(self.atlas_image_paths)
        payload["fetched_positions_mm"] = list(self.fetched_positions_mm)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> TerminalState:
        return cls(
            section_id=str(obj["section_id"]),
            subject_id=str(obj["subject_id"]),
            atlas_name=str(obj["atlas_name"]),
            plane=str(obj["plane"]),  # type: ignore[arg-type]
            valid_range_mm=(float(obj["valid_range_mm"][0]), float(obj["valid_range_mm"][1])),
            ground_truth_mm=float(obj["ground_truth_mm"]),
            query_image_path=str(obj["query_image_path"]),
            atlas_image_paths=tuple(str(p) for p in obj.get("atlas_image_paths", [])),
            fetched_positions_mm=tuple(float(p) for p in obj.get("fetched_positions_mm", [])),
            source=str(obj.get("source", "")),
            quality=dict(obj.get("quality", {})),
        )


def write_terminal_states(rows: Iterable[TerminalState], path: Path) -> int:
    """Write rows to ``path`` (creating parent dirs). Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.to_jsonl() + "\n")
            n += 1
    return n


def read_terminal_states(path: Path) -> Iterator[TerminalState]:
    """Yield ``TerminalState`` rows from a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            yield TerminalState.from_dict(json.loads(stripped))


_VALID_RANGE_CACHE: dict[tuple[str, Plane], tuple[float, float]] = {}


def _atlas_valid_range_mm(atlas_name: str, plane: Plane) -> tuple[float, float]:
    """Cached per (atlas, plane). Loading a BrainGlobe atlas is multi-second."""
    key = (atlas_name, plane)
    cached = _VALID_RANGE_CACHE.get(key)
    if cached is not None:
        return cached
    # Local import — keeps the module importable in unit tests that don't
    # have BrainGlobe atlases on disk.
    from langslice_harness.atlas.core import (  # noqa: PLC0415
        get_position_range_mm,
        load_atlas,
    )

    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    out = (float(pos_lo), float(pos_hi))
    _VALID_RANGE_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Manifest GT lookup
# ---------------------------------------------------------------------------


def _section_matches_stem(section_part: str, stem: str) -> bool:
    """Word-boundary match: the section part must appear delimited by ``_``
    or stem boundaries.

    Substring-only matching would false-positive on short numeric sections
    (``ad_bxd`` slices are numbered ``2``, ``5``, ``7`` — bare ``"7"`` would
    match many unrelated stems). Word-boundary matching keeps single-digit
    sections distinguishable while still accepting the multi-token sections
    common in ``deepslice_gt`` (e.g. ``641_2002_2565_NM01_s102_10x_M``).
    """
    if not section_part:
        return False
    pattern = r"(?:^|_)" + re.escape(section_part) + r"(?:_|$)"
    return re.search(pattern, stem) is not None


class ManifestGTLookup:
    """Index manifest shards by (plane, subject_id) for fast GT join.

    On first lookup for a plane, walks ``<manifest_root>/shards/<plane>/*.jsonl``
    and builds an in-memory ``{subject_id: [(dataset, section_id, position_mm)]}``
    index. Lookups then narrow to the SFT row's subject_id and find a
    candidate whose section-part appears in the query image filename's stem.

    The match strategy:
      1. Filter candidates to those with the SFT row's ``subject_id``.
      2. Within those, keep candidates whose section_id's after-colon part
         appears as a word-bounded token in the query stem.
      3. If exactly one candidate matches → return it.
      4. If multiple match (rare; possible when section parts coincidentally
         substring-overlap), break the tie by preferring the one whose
         ``dataset`` name appears in the query stem.
      5. If none match → return None and let the caller drop the row.
    """

    def __init__(self, manifest_root: Path) -> None:
        self._manifest_root = manifest_root
        self._cache: dict[str, dict[str, list[tuple[str, str, float]]]] = {}

    def _ensure_plane(self, plane: str) -> None:
        if plane in self._cache:
            return
        out: dict[str, list[tuple[str, str, float]]] = {}
        plane_dir = self._manifest_root / "shards" / plane
        if not plane_dir.is_dir():
            self._cache[plane] = out
            return
        # ``*.jsonl`` glob skips backup files (``.bak.<ts>.<reason>``) since
        # those don't end in ``.jsonl``.
        for shard_path in sorted(plane_dir.glob("*.jsonl")):
            dataset = shard_path.stem
            try:
                with shard_path.open(encoding="utf-8") as fh:
                    for raw in fh:
                        stripped = raw.strip()
                        if not stripped:
                            continue
                        row = json.loads(stripped)
                        sid = row.get("section_id")
                        subj = row.get("subject_id")
                        pos = row.get("position_mm")
                        if not sid or subj is None or pos is None:
                            continue
                        out.setdefault(str(subj), []).append(
                            (dataset, str(sid), float(pos))
                        )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("manifest GT: failed to read %s: %s", shard_path, exc)
                continue
        self._cache[plane] = out

    def lookup(
        self, plane: str, subject_id: str, query_image_stem: str,
    ) -> tuple[str, str, float] | None:
        """Return ``(dataset, section_id, position_mm)`` for the SFT row, or None."""
        self._ensure_plane(plane)
        candidates = self._cache[plane].get(subject_id, [])
        matched: list[tuple[str, str, float]] = []
        for dataset, sid, pos in candidates:
            section_part = sid.split(":", 1)[1] if ":" in sid else sid
            if _section_matches_stem(section_part, query_image_stem):
                matched.append((dataset, sid, pos))
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            # Tie-break: prefer the candidate whose dataset name appears in
            # the stem (deepslice_gt vs allen_dev_coronal both could carry
            # subject "NM01", but only one matches the stem prefix).
            for c in matched:
                if c[0] in query_image_stem:
                    return c
            # Still ambiguous — log and skip rather than guess.
            logger.warning(
                "manifest GT: ambiguous match for plane=%s subject=%s stem=%s: %s",
                plane, subject_id, query_image_stem,
                [(d, s) for d, s, _ in matched],
            )
            return None
        return None


# ---------------------------------------------------------------------------
# Trace walker
# ---------------------------------------------------------------------------


def _row_passes_tier(row: dict[str, Any], tier: TierFilter) -> bool:
    if tier == "any":
        return True
    quality = row.get("quality") or {}
    actual = str(quality.get("acceptance_tier", "")).strip()
    if tier == "strict":
        return actual == "strict"
    if tier == "rescued":
        return actual in {"strict", "rescued"}
    raise ValueError(f"unknown tier filter: {tier!r}")


def _walk_trace_to_terminal(
    trace: list[dict[str, Any]],
) -> tuple[list[str], list[float], dict[str, Any]] | None:
    """Return ``(atlas_image_paths, fetched_positions_mm, submit_args)`` or None.

    Walks ``trace`` collecting every ``fetch_atlas`` tool result's image paths
    in chronological order, parsing the snapped position from each filename
    so paths and positions agree by construction. Stops at the first
    ``submit_estimate`` step. Returns ``None`` if the trace never submits or
    if any atlas filename is unparseable (fail closed — never propagate
    misaligned captions).
    """
    image_paths: list[str] = []
    positions: list[float] = []
    for step in trace:
        if "submit" in step:
            submit = step["submit"]
            if not isinstance(submit, dict):
                return None
            args = submit.get("args")
            if not isinstance(args, dict):
                return None
            return image_paths, positions, args
        if "tool_call" in step:
            call = step.get("tool_call") or {}
            result = step.get("tool_result") or {}
            if call.get("name") != "fetch_atlas":
                continue
            paths = result.get("image_paths") or []
            if not isinstance(paths, list):
                continue
            for p in paths:
                p_str = str(p)
                pos = _position_mm_from_atlas_path(p_str)
                if pos is None:
                    # Unparseable atlas filename → can't trust this trace's
                    # captions. Better to drop the whole row than ship a
                    # mislabeled training pair.
                    return None
                image_paths.append(p_str)
                positions.append(pos)
    return None


def _query_stem_for_match(query_image_path: str) -> str:
    """Strip directory + extension + ``_clahe`` suffix for manifest matching."""
    stem = Path(query_image_path).stem
    if stem.endswith("_clahe"):
        stem = stem[: -len("_clahe")]
    return stem


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_from_sft_corpus(
    sft_corpus_path: Path,
    *,
    sft_corpus_root: Path | None = None,
    manifest_root: Path | None = None,
    tier: TierFilter = "strict",
    max_rows: int | None = None,
    require_query_on_disk: bool = True,
    require_atlas_on_disk: bool = True,
    repo_root: Path | None = None,
) -> Iterator[TerminalState]:
    """Walk the SFT JSONL and yield one TerminalState per accepted trace.

    Parameters
    ----------
    sft_corpus_path:
        Path to the SFT JSONL (one row per trace).
    sft_corpus_root:
        Directory paths inside the JSONL are relative to. Defaults to the
        JSONL's parent (matches ``models/langslice-gemma-4/data/``).
    manifest_root:
        Root of the data manifest used to resolve true GT. Defaults to
        ``data/manifest`` (canonical location per ``AGENTS.md``). Rows whose
        ``(plane, subject_id, query_stem)`` doesn't join to a manifest row
        are dropped — never silently fall back to the teacher's submit.
    tier:
        Acceptance-tier filter — ``"strict"`` (default), ``"rescued"``
        (strict + rescued), or ``"any"``.
    max_rows:
        Optional cap on emitted rows (useful for smoke runs).
    require_query_on_disk / require_atlas_on_disk:
        Drop rows whose referenced images are missing under
        ``sft_corpus_root``.
    repo_root:
        Used to compute the repo-relative path stored in the JSONL.
        Defaults to ``Path.cwd()`` so paths take the form
        ``models/langslice-gemma-4/data/queries/...`` when invoked from the
        repo root — matching the trainer's ``--repo-root .`` default.
    """
    if tier not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {_VALID_TIERS}, got {tier!r}")
    sft_root = sft_corpus_root or sft_corpus_path.parent
    base_for_paths = repo_root or Path.cwd()
    manifest_lookup = ManifestGTLookup(manifest_root or Path("data/manifest"))

    emitted = 0
    skipped_kind = 0
    skipped_tier = 0
    skipped_no_submit = 0
    skipped_missing_query = 0
    skipped_missing_atlas = 0
    skipped_no_manifest_gt = 0

    with sft_corpus_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("system_prompt_kind") != "single_slice":
                skipped_kind += 1
                continue
            if not _row_passes_tier(row, tier):
                skipped_tier += 1
                continue

            trace = row.get("trace") or []
            walked = _walk_trace_to_terminal(trace)
            if walked is None:
                skipped_no_submit += 1
                continue
            atlas_paths_rel, positions, submit_args = walked

            # Teacher's submit position — kept as a diagnostic only. Never
            # used as the training reward target.
            try:
                teacher_pos = float(submit_args["position_mm"])
            except (KeyError, TypeError, ValueError):
                teacher_pos = None

            query_paths = row.get("query_image_paths") or []
            if not query_paths:
                skipped_missing_query += 1
                continue
            query_rel = str(query_paths[0])
            query_abs = (sft_root / query_rel).resolve()
            if require_query_on_disk and not query_abs.is_file():
                logger.warning("query image missing on disk: %s", query_abs)
                skipped_missing_query += 1
                continue

            if require_atlas_on_disk:
                missing = [p for p in atlas_paths_rel if not (sft_root / p).is_file()]
                if missing:
                    logger.warning(
                        "atlas image(s) missing for %s: %s",
                        query_rel, missing[:3],
                    )
                    skipped_missing_atlas += 1
                    continue

            atlas_name = str(row["atlas_name"])
            plane: Plane = str(row["plane"])  # type: ignore[assignment]
            subject_id = str(row.get("subject_id", ""))

            # True-GT lookup. Drop rows that don't resolve — using the teacher
            # submit as a fallback would cap the policy at teacher-level
            # accuracy and turn RLVR into imitation.
            gt_lookup = manifest_lookup.lookup(
                plane, subject_id, _query_stem_for_match(query_rel),
            )
            if gt_lookup is None:
                skipped_no_manifest_gt += 1
                continue
            dataset_name, section_id, gt_mm = gt_lookup

            try:
                valid_range = _atlas_valid_range_mm(atlas_name, plane)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "could not resolve valid_range_mm for (%s, %s): %s",
                    atlas_name, plane, exc,
                )
                continue

            def _to_repo_rel(abs_or_rel: str | Path) -> str:
                p = Path(abs_or_rel)
                if not p.is_absolute():
                    p = (sft_root / p).resolve()
                try:
                    rel = p.resolve().relative_to(base_for_paths.resolve())
                except ValueError:
                    return p.as_posix()
                return rel.as_posix()

            quality = dict(row.get("quality") or {})
            if teacher_pos is not None:
                quality["teacher_position_mm"] = teacher_pos
            quality["dataset"] = dataset_name

            yield TerminalState(
                section_id=section_id,
                subject_id=subject_id,
                atlas_name=atlas_name,
                plane=plane,
                valid_range_mm=valid_range,
                ground_truth_mm=gt_mm,
                query_image_path=_to_repo_rel(query_rel),
                atlas_image_paths=tuple(_to_repo_rel(p) for p in atlas_paths_rel),
                fetched_positions_mm=tuple(positions),
                source=f"sft_corpus:{tier}",
                quality=quality,
            )
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                break

    print(
        f"[build_from_sft_corpus] emitted={emitted} skipped_kind={skipped_kind} "
        f"skipped_tier={skipped_tier} skipped_no_submit={skipped_no_submit} "
        f"skipped_missing_query={skipped_missing_query} "
        f"skipped_missing_atlas={skipped_missing_atlas} "
        f"skipped_no_manifest_gt={skipped_no_manifest_gt}",
        file=sys.stderr, flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cmd(args: argparse.Namespace) -> int:
    rows = build_from_sft_corpus(
        sft_corpus_path=args.sft_corpus,
        sft_corpus_root=args.sft_corpus_root,
        manifest_root=args.manifest_root,
        tier=args.tier,
        max_rows=args.max_rows,
        require_query_on_disk=not args.no_check_disk,
        require_atlas_on_disk=not args.no_check_disk,
        repo_root=args.repo_root,
    )
    n = write_terminal_states(rows, args.output)
    print(f"wrote {n} terminal-state rows to {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="single_turn_rl.terminal_states",
        description="Build terminal-state JSONL for single-turn GRPO (Lane A).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Walk an SFT corpus and emit a terminal-state JSONL.")
    pb.add_argument("--sft-corpus", type=Path, required=True,
                    help="Path to the SFT JSONL (e.g. data/sft_examples.jsonl).")
    pb.add_argument("--sft-corpus-root", type=Path, default=None,
                    help="Directory the JSONL's relative paths resolve against. "
                    "Defaults to the JSONL's parent.")
    pb.add_argument("--manifest-root", type=Path, default=Path("data/manifest"),
                    help="Root of the data manifest for true-GT lookup. "
                    "Defaults to data/manifest (canonical per AGENTS.md).")
    pb.add_argument("--output", type=Path, required=True,
                    help="Destination JSONL path.")
    pb.add_argument("--tier", choices=_VALID_TIERS, default="strict",
                    help="Acceptance-tier filter (default: strict).")
    pb.add_argument("--max-rows", type=int, default=None,
                    help="Optional cap for smoke runs.")
    pb.add_argument("--no-check-disk", action="store_true",
                    help="Skip image-on-disk existence checks.")
    pb.add_argument("--repo-root", type=Path, default=None,
                    help="Used to compute repo-relative paths in the JSONL. "
                    "Defaults to the current working directory so paths align "
                    "with the trainer's --repo-root . default.")
    pb.set_defaults(func=_build_cmd)

    ns = p.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main(sys.argv[1:]))
