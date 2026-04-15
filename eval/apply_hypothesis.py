"""Helper to apply and revert hypothesis changes to ap_multi_slice.py.

Usage:
    # Apply a prompt hypothesis
    python eval/apply_hypothesis.py apply-prompt h01

    # Revert to baseline
    python eval/apply_hypothesis.py revert
"""

from __future__ import annotations

import os
import re
import shutil
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TARGET = os.path.join(
    _REPO_ROOT, "langslice", "estimation", "google", "ap_multi_slice.py"
)
_BACKUP = _TARGET + ".baseline"
_HYPOTHESES_DIR = os.path.join(_REPO_ROOT, "eval", "hypotheses")


def _backup_baseline() -> None:
    """Save a copy of the baseline file if not already saved."""
    if not os.path.exists(_BACKUP):
        shutil.copy2(_TARGET, _BACKUP)
        print(f"Baseline saved to {_BACKUP}")
    else:
        print("Baseline backup already exists.")


def _revert() -> None:
    """Restore the baseline file."""
    if os.path.exists(_BACKUP):
        shutil.copy2(_BACKUP, _TARGET)
        print("Reverted to baseline.")
    else:
        print("No baseline backup found — nothing to revert.")


def _apply_prompt(hypothesis_id: str) -> None:
    """Replace the system_instruction in ap_multi_slice.py with the hypothesis prompt."""
    prompt_file = os.path.join(_HYPOTHESES_DIR, f"{hypothesis_id}_prompt.txt")
    if not os.path.exists(prompt_file):
        print(f"ERROR: {prompt_file} not found")
        sys.exit(1)

    # Read the hypothesis prompt (skip comment lines)
    with open(prompt_file, encoding="utf-8") as f:
        lines = f.readlines()
    # Find the actual prompt content (skip # comment lines)
    prompt_lines = []
    in_prompt = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("(") or in_prompt:
            in_prompt = True
            prompt_lines.append(line)
    prompt_content = "".join(prompt_lines).strip()
    # Remove outer parentheses
    if prompt_content.startswith("(") and prompt_content.endswith(")"):
        prompt_content = prompt_content[1:-1].strip()

    _backup_baseline()

    # Read current file
    with open(_TARGET, encoding="utf-8") as f:
        source = f.read()

    # Find and replace the system_instruction assignment
    # Pattern: system_instruction = (...multiline...)
    pattern = r'(    # --- System prompt ---\n    system_instruction = \().*?(\n    \))'
    replacement_text = f"    # --- System prompt ---\n    # [HYPOTHESIS: {hypothesis_id}]\n    system_instruction = (\n{_indent(prompt_content, 8)}\n    )"

    # Simpler approach: find the system_instruction block by markers
    start_marker = "    # --- System prompt ---"
    end_marker = "    # --- Loop setup ---"

    start_idx = source.find(start_marker)
    end_idx = source.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("ERROR: Could not find system prompt markers in target file")
        sys.exit(1)

    new_block = f"    # --- System prompt ---\n    # [HYPOTHESIS: {hypothesis_id}]\n    system_instruction = (\n{_indent(prompt_content, 8)}\n    )\n\n"

    new_source = source[:start_idx] + new_block + source[end_idx:]

    with open(_TARGET, "w", encoding="utf-8") as f:
        f.write(new_source)

    print(f"Applied prompt hypothesis: {hypothesis_id}")


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text by the given number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python eval/apply_hypothesis.py <command> [args]")
        print("Commands: apply-prompt <id>, revert")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "apply-prompt":
        if len(sys.argv) < 3:
            print("Usage: python eval/apply_hypothesis.py apply-prompt <hypothesis_id>")
            sys.exit(1)
        _apply_prompt(sys.argv[2])
    elif cmd == "revert":
        _revert()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
