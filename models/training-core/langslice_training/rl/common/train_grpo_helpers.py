"""Shared helper utilities for GRPO train/eval entrypoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _install_optional_dep_stubs() -> None:
    """Install lightweight stubs for optional TRL imports on non-Ascend hosts."""
    import importlib.machinery
    import sys
    import types

    def _stub_module(name: str, *, is_package: bool) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
        mod.__spec__ = spec
        if is_package:
            mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
        return mod

    if "vllm_ascend" not in sys.modules:
        _stub_module("vllm_ascend", is_package=True)
        _stub_module("vllm_ascend.distributed", is_package=True)
        _stub_module("vllm_ascend.distributed.device_communicators", is_package=True)
        pyhccl = _stub_module(
            "vllm_ascend.distributed.device_communicators.pyhccl", is_package=False
        )

        class _PyHcclStub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("vllm_ascend stub — not available on this hardware")

        pyhccl.PyHcclCommunicator = _PyHcclStub  # type: ignore[attr-defined]

    try:
        import trl.import_utils as _trl_iu  # noqa: PLC0415

        for _attr in list(vars(_trl_iu).keys()):
            if _attr.startswith("_") and _attr.endswith("_available"):
                _val = getattr(_trl_iu, _attr)
                if isinstance(_val, tuple):
                    setattr(_trl_iu, _attr, bool(_val[0]))
        for _attr in ("_vllm_available", "_vllm_ascend_available"):
            if hasattr(_trl_iu, _attr):
                setattr(_trl_iu, _attr, False)
    except (ImportError, AttributeError):
        pass


def _filter_grpo_config_for_installed_trl(
    grpo_config_cls: type,
    grpo_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Drop config keys unsupported by the installed GRPOConfig class."""
    import inspect

    accepted: set[str] | None = None
    try:
        mro = inspect.getmro(grpo_config_cls)
    except (AttributeError, TypeError):
        return grpo_cfg

    for cls in mro:
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            continue
        accepted = {p.name for p in params if p.name != "self"}
        break

    if accepted is None:
        return grpo_cfg

    out: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in grpo_cfg.items():
        if key in accepted:
            out[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.warning(
            "Dropping unsupported GRPOConfig key(s) for installed TRL: %s",
            ", ".join(sorted(dropped)),
        )
    return out


def _adapter_base_model_name(adapter_dir: Path) -> str | None:
    """Return the PEFT adapter's base model id, or None for non-adapter paths."""
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        return None
    try:
        config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PEFT adapter config: {adapter_config_path}") from exc
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(
            f"PEFT adapter config missing base_model_name_or_path: {adapter_config_path}"
        )
    return base_model

