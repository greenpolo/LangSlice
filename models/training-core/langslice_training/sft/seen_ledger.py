"""Persistent seen-section ledger for SFT curation workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

__all__ = ["SeenLedger", "StaleLedgerError"]


class StaleLedgerError(RuntimeError):
    """Raised when the ledger's manifest fingerprint no longer matches disk."""


class SeenLedger:
    """Persistent append-only ledger of consumed section_ids."""

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

    def load(self) -> None:
        if not self._path.is_file():
            self._header_written = False
            return

        current_fp = self._compute_fingerprint()
        with self._path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        if not lines:
            self._header_written = False
            return

        header_raw = lines[0].strip()
        if not header_raw:
            self._header_written = False
            return

        try:
            header = json.loads(header_raw)
        except json.JSONDecodeError as exc:
            raise StaleLedgerError(
                f"Ledger header at {self._path} is not valid JSON: {exc}. "
                "Delete the file or reset the ledger."
            ) from exc

        if not header.get("_header"):
            raise StaleLedgerError(
                f"Ledger at {self._path} does not start with a valid header. "
                "Delete the file or reset the ledger."
            )

        stored_fp = header.get("fingerprint", "")
        if stored_fp != current_fp:
            raise StaleLedgerError(
                f"Manifest changed since ledger write "
                f"(stored={stored_fp!r}, current={current_fp!r})."
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
        self._seen.add(section_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._header_written:
            self._write_header()
        entry = json.dumps(
            {"round_idx": round_idx, "section_id": section_id, "source": source},
            ensure_ascii=False,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

    def is_seen(self, section_id: str) -> bool:
        return section_id in self._seen

    def seen_count(self) -> int:
        return len(self._seen)

    @property
    def seen(self) -> set[str]:
        return self._seen

    def _write_header(self) -> None:
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
        mode = "a"
        if not self._path.exists() or self._path.stat().st_size == 0:
            mode = "w"
        with self._path.open(mode, encoding="utf-8") as fh:
            fh.write(header + "\n")
        self._header_written = True

    def _compute_fingerprint(self) -> str:
        if self._manifest_root is None or not self._manifest_root.is_dir():
            return "no-manifest"

        entries: list[tuple[str, int, int]] = []
        for plane in self._PLANES:
            for glob_pattern, suffix in (("shards", "*.jsonl"), ("overrides", "*.json")):
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

        entries.sort()
        digest = hashlib.sha256()
        for rel, mtime, size in entries:
            digest.update(f"{rel}\x00{mtime}\x00{size}\n".encode())
        return digest.hexdigest()

__all__ = ["SeenLedger", "StaleLedgerError"]
