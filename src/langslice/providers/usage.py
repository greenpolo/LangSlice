"""Content-free usage diagnostics; never changes the provider request.

The subscription backend's per-item attribution is optional and undocumented.
Generated item identifiers are not request indices: only explicit ID matches
are mapped. Parent counters and child content counters must not be added twice.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

COUNTERS = ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens")
_ITEM_TYPES = {"message", "reasoning", "function_call", "function_call_output"}
_CONTENT_TYPES = {"input_text", "output_text", "input_image", "input_file"}


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _counters(value: Any) -> dict[str, int | None]:
    data = value if isinstance(value, dict) else {}
    return {key: _count(data.get(key)) for key in COUNTERS}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def item_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    """Only types, hashed identities and hashed content; no payload or URLs."""
    kind = item.get("type", "message")
    result: dict[str, Any] = {
        "type": kind if kind in _ITEM_TYPES else "other",
        "fingerprint": _digest(item),
    }
    for key in ("id", "call_id"):
        if isinstance(item.get(key), str):
            result[key + "_sha256"] = _digest(item[key])
    if item.get("role") in {"user", "assistant", "system", "developer"}:
        result["role"] = item["role"]
    parts = item.get("output") if kind == "function_call_output" else item.get("content")
    if isinstance(parts, list):
        result["content"] = [
            {
                "index": index,
                "type": part.get("type") if isinstance(part, dict)
                and part.get("type") in _CONTENT_TYPES else "other",
                "fingerprint": _digest(part),
            }
            for index, part in enumerate(parts)
        ]
    return result


def request_descriptor(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [item_descriptor(item) for item in body.get("input", [])],
        "fields": {key: _digest(body[key]) for key in ("instructions", "tools") if key in body},
    }


def _residual(
    total: dict[str, int | None], rows: list[dict[str, int | None]],
) -> dict[str, int | None]:
    return {
        key: total[key] - sum(row[key] for row in rows)  # type: ignore[misc]
        if total[key] is not None and all(row[key] is not None for row in rows) else None
        for key in COUNTERS
    }


def usage_diagnostics(
    raw: Any, request: dict[str, Any] | None = None,
    outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Allowlisted counts plus optional observational attribution, unknowns intact."""
    raw = raw if isinstance(raw, dict) else {}
    details = raw.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    output_details = raw.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    totals = _counters({**raw, "cached_tokens": details.get("cached_tokens"),
                        "cache_write_tokens": details.get("cache_write_tokens")})
    result: dict[str, Any] = {
        "schema_version": 1,
        "totals": {**totals, "reasoning_tokens": _count(output_details.get("reasoning_tokens")),
                   "total_tokens": _count(raw.get("total_tokens"))},
    }
    if request is None:
        return result
    result["request"] = request
    result["outputs"] = outputs or []
    attribution = raw.get("attribution")
    if not isinstance(attribution, dict):
        result["attribution_status"] = "unavailable"
        return result
    incoming = attribution.get("items")
    fields = attribution.get("request_fields")
    if not isinstance(incoming, dict) or not isinstance(fields, dict):
        result["attribution_status"] = "malformed"
        return result
    items = []
    for identifier, data in incoming.items():
        row: dict[str, Any] = {"id_sha256": _digest(identifier), "tokens": _counters(data)}
        matches = [index for index, item in enumerate(request["items"])
                   if item.get("id_sha256") == row["id_sha256"]]
        output_matches = [index for index, item in enumerate(outputs or [])
                          if item.get("id_sha256") == row["id_sha256"]]
        row["mapping"] = "unmapped"
        if len(matches) == 1 and not output_matches:
            row.update(mapping="exact_request_id", request_index=matches[0])
        elif len(output_matches) == 1 and not matches:
            row.update(mapping="exact_output_id", output_index=output_matches[0])
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            row["content"] = [_counters(part) for part in data["content"]]
            row["content_residual"] = _residual(row["tokens"], row["content"])
        items.append(row)
    # Unknown field names might contain content: hash rather than copying them.
    field_rows = [{"field": key if key in {"tools", "instructions"} else "other",
                   "field_sha256": _digest(key), "tokens": _counters(data)}
                  for key, data in fields.items()]
    residual = _residual(totals, [row["tokens"] for row in [*items, *field_rows]])
    result.update(
        attribution_status="observed_optional",
        attribution={"items": items, "request_fields": field_rows},
        attribution_residual=residual,
        attribution_reconciled=all(value == 0 for value in residual.values()),
    )
    return result
