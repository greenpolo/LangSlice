"""--stage submit: pack verified examples, submit to AI Studio Batch API,
poll, retrieve, mm-strip, write final SFT."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "models" / "langslice-gemma-4" / "data"))

import bbox_io  # noqa: E402

log = logging.getLogger("stage_submit")

_POLL_INTERVAL_S = 30.0


def _get_genai_client():
    """Return the configured google-genai client for Batch API calls.

    Test stubs expose `.files.upload`, `.batches.create`, `.batches.get`, and
    `.files.download`, matching the current Gemini Batch API flow.
    """
    from langslice_harness import vlm_config
    if not vlm_config.supports_batch_api():
        raise RuntimeError(
            f"Backend {vlm_config.get_backend()!r} does not support Batch API"
        )
    return vlm_config.get_client()


def _encode_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_request_jsonl(
    draft: list[dict], verdicts: dict[str, str], out_path: Path
) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in draft:
            if verdicts.get(record["id"]) != "verify":
                continue
            images_b64 = [
                _encode_b64(REPO_ROOT / p) for p in record["section_image_paths"]
            ]
            line = bbox_io.pack_batch_request(record, images_b64=images_b64)
            f.write(json.dumps(line) + "\n")
            n += 1
    return n


def _read_verdicts(path: Path) -> dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec["verdict"]
    return out


def _poll_until_done(client, batch_name: str) -> object:
    while True:
        info = client.batches.get(name=batch_name)
        state = info.get("state") if isinstance(info, dict) else getattr(info, "state", None)
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            log.info("Batch %s reached state %s", batch_name, state)
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch {batch_name} ended in state {state}")
            return info
        log.info("Batch %s state=%s; sleeping %.0fs", batch_name, state, _POLL_INTERVAL_S)
        time.sleep(_POLL_INTERVAL_S)


def _batch_dest_file_name(batch_job: object) -> str:
    if isinstance(batch_job, dict):
        return str(batch_job["dest"]["file_name"])
    dest = batch_job.dest
    return str(dest.file_name)


def _build_sft_record(draft_record: dict, caption: str) -> dict:
    user_text = bbox_io.build_user_text(draft_record)
    image_parts = [
        {"type": "image", "path": p} for p in draft_record["section_image_paths"]
    ]
    return {
        "id": draft_record["id"],
        "atlas": draft_record["atlas"],
        "atlas_version": draft_record["atlas_version"],
        "orientation": draft_record["orientation"],
        "region": draft_record["region"],
        "source_type": draft_record["source_type"],
        "source_brain": draft_record["source_brain"],
        "is_hemisphere": draft_record["is_hemisphere"],
        "messages": [
            {"role": "system", "content": bbox_io._SYSTEM_PROMPT},
            {"role": "user", "content": image_parts + [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": caption},
        ],
    }


def run_stage_submit(args: argparse.Namespace) -> int:
    draft_path = args.out_dir / "draft_manifest.jsonl"
    if not draft_path.exists():
        log.error("Draft manifest missing: %s", draft_path)
        return 1
    draft = list(bbox_io.read_jsonl(draft_path))

    verdicts = _read_verdicts(args.verdicts)
    log.info("Read %d verdicts; %d verified", len(verdicts),
             sum(1 for v in verdicts.values() if v == "verify"))

    request_path = args.out_dir / "request.jsonl"
    n = _build_request_jsonl(draft, verdicts, request_path)
    log.info("Built %d batch requests at %s", n, request_path)
    if n == 0:
        log.error("No verified examples — nothing to submit")
        return 1

    if args.dry_run:
        log.info("--dry-run: skipping API submission")
        return 0

    client = _get_genai_client()

    # Upload the JSONL file.
    uploaded_file = client.files.upload(
        file=str(request_path),
        config={"display_name": f"bbox-{int(time.time())}"},
    )
    log.info("Uploaded file %s", uploaded_file.name)

    batch = client.batches.create(model=args.model, src=uploaded_file.name)
    batch_name = batch.get("name") if isinstance(batch, dict) else getattr(batch, "name", None)
    log.info("Created batch %s", batch_name)

    metadata_path = args.out_dir / "batch_metadata.json"
    metadata_path.write_text(
        json.dumps({"batch_name": batch_name, "model": args.model,
                    "submitted_at": time.time(), "n_requests": n}, indent=2),
        encoding="utf-8",
    )

    completed_batch = _poll_until_done(client, batch_name)

    # Pull responses from the result file named by batch_job.dest.file_name.
    result_file_name = _batch_dest_file_name(completed_batch)
    downloaded = client.files.download(file=result_file_name)
    response_text = downloaded.decode("utf-8") if isinstance(downloaded, bytes) else str(downloaded)
    responses_path = args.out_dir / "responses.jsonl"
    by_id: dict[str, dict] = {}
    with responses_path.open("w", encoding="utf-8") as f:
        for line in response_text.splitlines():
            if not line.strip():
                continue
            resp = json.loads(line)
            f.write(json.dumps(resp) + "\n")
            by_id[resp["key"]] = resp

    # Build SFT records (mm-strip + length gate).
    sft_records: list[dict] = []
    n_dropped = 0
    for record in draft:
        if verdicts.get(record["id"]) != "verify":
            continue
        resp = by_id.get(record["id"])
        if resp is None:
            n_dropped += 1
            continue
        caption = bbox_io.extract_caption_from_response(resp.get("response", {}))
        if caption is None:
            n_dropped += 1
            continue
        if not bbox_io.length_gate(caption):
            n_dropped += 1
            continue
        if bbox_io.mm_strip_filter(caption) is None:
            n_dropped += 1
            continue
        sft_records.append(_build_sft_record(record, caption))

    bbox_io.write_jsonl(args.out_dir / "sft.jsonl", sft_records)
    log.info("Wrote %d SFT records (%d dropped) to sft.jsonl", len(sft_records), n_dropped)
    return 0
