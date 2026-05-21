"""Tests for the ``SubmitEstimateCloseCriteria`` + post-generate eos rewrite.

These exercise the two halves of the TRL ``mask_truncated_completions`` fix
in ``single_turn_rl.train_grpo`` *without* loading the model:

1. ``_find_submit_close_index`` -- pure-Python string-aware brace counter.
2. ``_completion_close_token_index`` + ``rewrite_completions_with_eos_after_submit_close``
   -- token-level walk + in-place eos/pad write, tested against the real
   Gemma 4 tokenizer at ``/models/sft-base``. Falls back to a tiny stub
   tokenizer when the real one is not on disk so the test still runs in
   CI-only environments.
3. ``_build_submit_estimate_stopping_criteria`` -- per-row latch behaviour
   over simulated multi-step generate calls.

Run via::

    docker exec langslice-training-dev bash -c \\
        "cd /workspace/LangSlice && python -m pytest \\
         tests/test_stop_after_submit_close.py -v"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from single_turn_rl.train_grpo import (
    SUBMIT_MARKER,
    _build_submit_estimate_stopping_criteria,
    _completion_close_token_index,
    _find_submit_close_index,
    rewrite_completions_with_eos_after_submit_close,
)

# --- Section 1: pure string-level brace counter ----------------------------


def test_find_submit_close_simple() -> None:
    text = "noise call:submit_estimate{position_mm: 5.0} more"
    end = _find_submit_close_index(text)
    assert end is not None
    # Slice up-to-and-including the close brace.
    assert text[:end].endswith("submit_estimate{position_mm: 5.0}")


def test_find_submit_close_skips_brace_inside_quoted_reasoning() -> None:
    # The first `}` is INSIDE a quoted string "a }brain"; the brace counter
    # MUST ignore it and land on the outer one after `chatter`'s preceding `}`.
    text = (
        'reasoning text ... '
        'call:submit_estimate{position_mm: 5.0, reasoning: "a }brain"} then chatter'
    )
    end = _find_submit_close_index(text)
    assert end is not None
    closed = text[:end]
    # The outer call must include the entire reasoning string and its close.
    assert closed.endswith('"a }brain"}')
    # And it must NOT have stopped at the inner `}`.
    assert not closed.endswith('"a }')


def test_find_submit_close_handles_escaped_quote() -> None:
    # Inside a quoted reasoning, an escaped quote should not end the string;
    # the brace after the (true) end-quote is the one that closes args.
    text = r'call:submit_estimate{r: "she said \"hi\" }", p: 3} tail'
    end = _find_submit_close_index(text)
    assert end is not None
    closed = text[:end]
    # The first `}` inside the quoted string must be ignored.
    assert closed.endswith('p: 3}')


def test_find_submit_close_no_marker_returns_none() -> None:
    # `{atlas_name}` appears but NOT preceded by the submit_estimate marker.
    text = "thinking about {atlas_name} the answer is unknown"
    assert _find_submit_close_index(text) is None


def test_find_submit_close_unclosed_returns_none() -> None:
    # Marker present but args dict still open.
    text = "call:submit_estimate{position_mm: 5.0, reasoning: \"thinking"
    assert _find_submit_close_index(text) is None


def test_find_submit_close_nested_braces_in_args() -> None:
    # Nested dict-like literal inside args should balance correctly.
    text = "call:submit_estimate{position_mm: 5.0, meta: {a: 1, b: 2}} tail"
    end = _find_submit_close_index(text)
    assert end is not None
    assert text[:end].endswith("{a: 1, b: 2}}")


# --- Section 2: tokenizer-level rewrite ------------------------------------


_GEMMA_PATH = Path("/models/sft-base")


def _load_tokenizer() -> Any:
    """Try to load the real Gemma 4 tokenizer; otherwise fall back to a stub.

    The stub mirrors only the API surface our code uses: ``decode``,
    ``eos_token_id``, ``pad_token_id``. It treats each "token" as a single
    character so token indices align trivially with character indices.
    """
    if _GEMMA_PATH.exists():
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(_GEMMA_PATH))
        except (OSError, ValueError, ImportError):
            pass

    class _CharTokenizer:
        eos_token_id = 1
        pad_token_id = 0

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:  # noqa: ARG002
            out_chars: list[str] = []
            for tok in ids:
                if tok == self.eos_token_id:
                    out_chars.append("")  # placeholder
                elif tok == self.pad_token_id:
                    out_chars.append("")
                else:
                    # Map ints to printable chars; tests build ids via _encode below.
                    out_chars.append(chr(tok))
            return "".join(out_chars)

        def encode_to_ids(self, text: str) -> list[int]:
            return [ord(c) for c in text]

    return _CharTokenizer()


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode using whichever tokenizer fixture is active.

    The real Gemma tokenizer is used via ``__call__``; the stub via its
    helper. We deliberately use ``add_special_tokens=False`` so prompt and
    completion concatenate without spurious bos.
    """
    if hasattr(tokenizer, "encode_to_ids"):
        return tokenizer.encode_to_ids(text)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return _load_tokenizer()


def test_completion_close_token_index_basic(tokenizer: Any) -> None:
    text = "reasoning then call:submit_estimate{position_mm: 5.0} chatter"
    ids = _encode(tokenizer, text)
    k = _completion_close_token_index(ids, tokenizer)
    assert k is not None
    # Decoded prefix up to and including token k must cover the close brace.
    decoded = tokenizer.decode(ids[: k + 1], skip_special_tokens=False)
    assert "}" in decoded
    assert SUBMIT_MARKER in decoded
    # The next character after the close should NOT be in the prefix (or be
    # whitespace at most). i.e. the prefix should END at or just past the `}`.
    char_pos = decoded.rfind("}")
    assert char_pos != -1
    assert len(decoded) - char_pos <= 4  # tolerate a token boundary slop of a few chars


def test_completion_close_token_index_ignores_inner_brace(tokenizer: Any) -> None:
    # The `}` inside the quoted reasoning must NOT trigger the close.
    text = 'call:submit_estimate{position_mm: 5.0, reasoning: "a }brain"} more'
    ids = _encode(tokenizer, text)
    k = _completion_close_token_index(ids, tokenizer)
    assert k is not None
    decoded = tokenizer.decode(ids[: k + 1], skip_special_tokens=False)
    # The decoded prefix must extend past the END of the reasoning string,
    # NOT stop at the inner brace.
    assert '"a }brain"' in decoded


def test_completion_close_token_index_no_marker(tokenizer: Any) -> None:
    text = "thinking about {atlas_name} the answer is unknown"
    ids = _encode(tokenizer, text)
    assert _completion_close_token_index(ids, tokenizer) is None


def test_rewrite_writes_eos_and_pad(tokenizer: Any) -> None:
    completion_text = "call:submit_estimate{position_mm: 5.0} TAIL_NOISE_TO_BE_PADDED"
    prompt_ids = _encode(tokenizer, "PROMPT ")
    comp_ids = _encode(tokenizer, completion_text)
    full = prompt_ids + comp_ids
    pcid = torch.tensor([full], dtype=torch.long)

    out = rewrite_completions_with_eos_after_submit_close(
        pcid, prompt_length=len(prompt_ids), tokenizer=tokenizer,
    )

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    completion = out[0, len(prompt_ids):]

    # The last token must now be eos or pad so TRL's truncation check passes.
    last = int(completion[-1].item())
    assert last in (eos, pad)

    # Exactly one eos should sit between the original close-brace and the
    # padding tail; everything after the eos position should be pad.
    eos_positions = (completion == eos).nonzero(as_tuple=True)[0].tolist()
    assert len(eos_positions) >= 1
    first_eos = eos_positions[0]
    if first_eos + 1 < completion.size(0):
        tail = completion[first_eos + 1 :]
        assert torch.all(tail == pad), f"expected all-pad tail, got {tail.tolist()}"


def test_rewrite_no_op_on_already_eos_terminated(tokenizer: Any) -> None:
    prompt_ids = _encode(tokenizer, "PROMPT ")
    comp_ids = _encode(tokenizer, "thinking about something")
    eos = tokenizer.eos_token_id
    comp_ids = comp_ids + [eos]  # naturally terminated
    full = prompt_ids + comp_ids
    pcid_before = torch.tensor([full], dtype=torch.long)
    pcid = pcid_before.clone()

    rewrite_completions_with_eos_after_submit_close(
        pcid, prompt_length=len(prompt_ids), tokenizer=tokenizer,
    )

    # Already-eos-terminated rows must be left untouched.
    assert torch.equal(pcid, pcid_before)


def test_rewrite_no_op_when_marker_absent(tokenizer: Any) -> None:
    prompt_ids = _encode(tokenizer, "PROMPT ")
    # No submit_estimate marker -- TRL will mask this as truncated, which is
    # the desired behavior. Our rewrite must NOT inject a fake eos.
    comp_ids = _encode(tokenizer, "thinking about {atlas_name} unsure")
    full = prompt_ids + comp_ids
    pcid_before = torch.tensor([full], dtype=torch.long)
    pcid = pcid_before.clone()

    rewrite_completions_with_eos_after_submit_close(
        pcid, prompt_length=len(prompt_ids), tokenizer=tokenizer,
    )

    assert torch.equal(pcid, pcid_before)


def test_rewrite_handles_quoted_close_brace(tokenizer: Any) -> None:
    # The brace counter must skip the `}` inside the reasoning string; the
    # eos rewrite must therefore land AFTER the true close brace.
    completion_text = (
        'call:submit_estimate{position_mm: 5.0, reasoning: "a }brain"} chatter'
    )
    prompt_ids = _encode(tokenizer, "P ")
    comp_ids = _encode(tokenizer, completion_text)
    pcid = torch.tensor([prompt_ids + comp_ids], dtype=torch.long)

    rewrite_completions_with_eos_after_submit_close(
        pcid, prompt_length=len(prompt_ids), tokenizer=tokenizer,
    )

    completion = pcid[0, len(prompt_ids):]
    eos = tokenizer.eos_token_id
    eos_positions = (completion == eos).nonzero(as_tuple=True)[0].tolist()
    assert len(eos_positions) >= 1
    first_eos = eos_positions[0]
    # The text BEFORE the eos must still include the full quoted reasoning.
    pre_eos_text = tokenizer.decode(
        completion[:first_eos].tolist(), skip_special_tokens=False,
    )
    assert '"a }brain"' in pre_eos_text


# --- Section 3: StoppingCriteria latch -------------------------------------


def test_stopping_criteria_fires_when_balanced(tokenizer: Any) -> None:
    """StoppingCriteria returns a per-row BoolTensor that latches True once
    a row emits a balanced ``submit_estimate{...}``."""
    prompt_text = "PROMPT "
    completion_text = "thinking call:submit_estimate{position_mm: 5.0} chatter"
    prompt_ids = _encode(tokenizer, prompt_text)
    comp_ids = _encode(tokenizer, completion_text)

    criterion = _build_submit_estimate_stopping_criteria(
        tokenizer, prompt_length=len(prompt_ids), decode_tail_tokens=128,
    )

    # Walk one token at a time; the criterion should flip True at or shortly
    # after the close-brace token.
    fired_at: int | None = None
    for k in range(1, len(comp_ids) + 1):
        full = prompt_ids + comp_ids[:k]
        input_ids = torch.tensor([full], dtype=torch.long)
        dummy_scores = torch.zeros((1, 1))
        result = criterion(input_ids, dummy_scores)
        assert result.shape == (1,)
        assert result.dtype == torch.bool
        if bool(result[0].item()) and fired_at is None:
            fired_at = k
            break

    assert fired_at is not None, "StoppingCriteria never fired"
    # Once fired, subsequent calls should stay fired (latch). Extend by 5
    # tokens of pad and confirm the latch holds.
    pad = tokenizer.pad_token_id
    extended = prompt_ids + comp_ids[:fired_at] + [pad] * 5
    input_ids = torch.tensor([extended], dtype=torch.long)
    dummy_scores = torch.zeros((1, 1))
    assert bool(criterion(input_ids, dummy_scores)[0].item())


def test_stopping_criteria_does_not_fire_on_inner_braces(tokenizer: Any) -> None:
    # `{atlas_name}` -- no submit_estimate marker -> must NOT fire.
    prompt_ids = _encode(tokenizer, "PROMPT ")
    comp_ids = _encode(tokenizer, "thinking about {atlas_name} the answer is unknown")

    criterion = _build_submit_estimate_stopping_criteria(
        tokenizer, prompt_length=len(prompt_ids), decode_tail_tokens=128,
    )

    for k in range(1, len(comp_ids) + 1):
        full = prompt_ids + comp_ids[:k]
        input_ids = torch.tensor([full], dtype=torch.long)
        dummy_scores = torch.zeros((1, 1))
        result = criterion(input_ids, dummy_scores)
        assert not bool(result[0].item()), (
            f"Criterion fired at token {k} on a marker-free completion"
        )


def test_stopping_criteria_per_row_independence(tokenizer: Any) -> None:
    """With batch_size > 1, only rows that have actually emitted a balanced
    submit_estimate should latch True; others stay False."""
    prompt_ids = _encode(tokenizer, "PROMPT ")
    finished_comp = _encode(
        tokenizer, "call:submit_estimate{position_mm: 5.0} chatter chatter",
    )
    unfinished_comp = _encode(
        tokenizer, "thinking about {atlas_name} still working",
    )

    # Pad to equal length so we can stack into a single (2, L) tensor.
    L = max(len(finished_comp), len(unfinished_comp))
    pad = tokenizer.pad_token_id
    finished_padded = finished_comp + [pad] * (L - len(finished_comp))
    unfinished_padded = unfinished_comp + [pad] * (L - len(unfinished_comp))

    batch = torch.tensor(
        [prompt_ids + finished_padded, prompt_ids + unfinished_padded],
        dtype=torch.long,
    )

    criterion = _build_submit_estimate_stopping_criteria(
        tokenizer, prompt_length=len(prompt_ids), decode_tail_tokens=128,
    )

    dummy_scores = torch.zeros((2, 1))
    result = criterion(batch, dummy_scores)
    assert result.shape == (2,)
    assert result.dtype == torch.bool
    assert bool(result[0].item()) is True, "row 0 (balanced) should fire"
    assert bool(result[1].item()) is False, "row 1 (no marker) should NOT fire"
