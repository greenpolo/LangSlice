"""Persistent seen-section ledger for expert-iteration SFT.

Tracks which section_ids have been consumed across iSFT rounds so the
bin-aware sampler v2 can avoid re-using the same sections (breadth over
depth).

On-disk format: newline-delimited JSON.
- Line 0: header ``{"_header": true, "fingerprint": "...", ...}``
- Lines 1+: entries ``{"round_idx": k, "section_id": "...", "source": "..."}``

Fingerprint: SHA-256 of (relative_path, mtime_seconds, file_size) for
every shard and override file under ``manifest_root``.  Mismatch on load
raises ``StaleLedgerError`` so the caller can reset or investigate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

__all__ = ["SeenLedger", "StaleLedgerError"]


class StaleLedgerError(RuntimeError):
    """Raised when the ledger's manifest fingerprint no longer matches the
    current manifest on disk.  The caller should either delete the ledger
    or pass ``--reset-ledger`` to iterate.py to start fresh.
    """


class SeenLedger:
    """Persistent append-only ledger of consumed section_ids.

    Parameters
    ----------
    ledger_path:
        JSONL file to read from / append to.
    manifest_root:
        Optional path to the manifest root (the directory containing
        ``shards/`` and ``overrides/`` subdirectories).  Used for the
        manifest-fingerprint check.  When ``None``, fingerprint is the
        degenerate ``"no-manifest"`` string (always matches itself).
    """

    _PLANES = ("coronal", "sagittal", "horizontal")

    def __init__(
        self,
        ledger_path: Path,
        manifest_root: Path | None = None,
    ) -> None:
        self._path = ledger_path
        self._manifest_root = manifest_root
        self._seen: set[str] = set()
        self._header_written = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the ledger from disk.

        - If the file doesn't exist, treat as an empty ledger (no error).
        - If the file exists, read the header and verify the manifest
          fingerprint.  Raises ``StaleLedgerError`` on mismatch.
        - Subsequent lines are entry rows; their ``section_id`` values
          are added to the in-memory seen set.
        """
        if not self._path.is_file():
            self._header_written = False
            return

        current_fp = self._compute_fingerprint()

        with self._path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        if not lines:
            self._header_written = False
            return

        # First non-empty line must be the header.
        header_raw = lines[0].strip()
        if not header_raw:
            self._header_written = False
            return

        try:
            header = json.loads(header_raw)
        except json.JSONDecodeError as exc:
            raise StaleLedgerError(
                f"Ledger header at {self._path} is not valid JSON: {exc}. "
                "Delete the file or pass --reset-ledger to start fresh."
            ) from exc

        if not header.get("_header"):
            raise StaleLedgerError(
                f"Ledger at {self._path} does not start with a valid header. "
                "Delete the file or pass --reset-ledger to start fresh."
            )

        stored_fp = header.get("fingerprint", "")
        if stored_fp != current_fp:
            raise StaleLedgerError(
                f"Manifest has changed since ledger was written "
                f"(fingerprint mismatch: stored={stored_fp!r}, "
                f"current={current_fp!r}). "
                f"Reset the ledger by deleting {self._path} or run "
                "iterate.py with --reset-ledger to start fresh."
            )

        self._header_written = True

        for raw in lines[1:]:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            sid = entry.get("section_id")
            if sid:
                self._seen.add(str(sid))

    def mark_seen(
        self,
        section_id: str,
        round_idx: int,
        source: Literal["synthetic", "real_rollout"],
    ) -> None:
        """Mark ``section_id`` as seen and append an entry to the JSONL.

        Idempotent in-memory (re-marking an already-seen id is a no-op for
        the set), but we still append the entry line so the on-disk log is
        a full audit trail.
        """
        self._seen.add(section_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write header on first append if file is empty / doesn't exist.
        if not self._header_written:
            self._write_header()
        entry = json.dumps(
            {"round_idx": round_idx, "section_id": section_id, "source": source},
            ensure_ascii=False,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

    def is_seen(self, section_id: str) -> bool:
        """Return True if ``section_id`` is in the in-memory seen set."""
        return section_id in self._seen

    def seen_count(self) -> int:
        """Return the number of distinct section_ids seen so far."""
        return len(self._seen)

    @property
    def seen(self) -> set[str]:
        """In-memory set of seen section_ids (read-only view)."""
        return self._seen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        """Write the header line if the file is empty or doesn't exist."""
        fp = self._compute_fingerprint()
        header = json.dumps(
            {
                "_header": True,
                "fingerprint": fp,
                "manifest_root": str(self._manifest_root) if self._manifest_root else None,
                "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        # Use "w" if file does not exist or is empty; otherwise "a" is safe
        # because we only call this when _header_written is False.
        mode = "a"
        if not self._path.exists() or self._path.stat().st_size == 0:
            mode = "w"
        with self._path.open(mode, encoding="utf-8") as fh:
            fh.write(header + "\n")
        self._header_written = True

    def _compute_fingerprint(self) -> str:
        """Compute a SHA-256 fingerprint of the manifest shard + override files.

        Walks ``manifest_root/shards/<plane>/*.jsonl`` and
        ``manifest_root/overrides/<plane>/*.json`` for all three planes.
        Collects ``(relative_path, mtime_seconds, file_size)`` tuples,
        sorts them, and hashes via SHA-256.

        Returns ``"no-manifest"`` when ``manifest_root`` is None or doesn't
        exist (degenerate but valid — always matches itself).
        """
        if self._manifest_root is None or not self._manifest_root.is_dir():
            return "no-manifest"

        entries: list[tuple[str, int, int]] = []
        for plane in self._PLANES:
            for glob_pattern, suffix in (
                ("shards", "*.jsonl"),
                ("overrides", "*.json"),
            ):
                target_dir = self._manifest_root / glob_pattern / plane
                if not target_dir.is_dir():
                    continue
                for p in sorted(target_dir.glob(suffix)):
                    try:
                        st = p.stat()
                        rel = str(p.relative_to(self._manifest_root))
                        entries.append((rel, int(st.st_mtime), int(st.st_size)))
                    except OSError:
                        continue

        # Sort for determinism across OS directory orderings.
        entries.sort()
        digest = hashlib.sha256()
        for rel, mtime, size in entries:
            digest.update(f"{rel}\x00{mtime}\x00{size}\n".encode())
        return digest.hexdigest()
