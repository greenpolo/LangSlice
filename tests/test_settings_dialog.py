"""Checks for settings dialog transport toggles."""

from __future__ import annotations

from pathlib import Path

import dotenv
from PySide6.QtWidgets import QApplication

import langslice.gui.settings_dialog as settings_dialog


def test_settings_dialog_loads_and_saves_transport_toggles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for key in (
        "LANGSLICE_GENAI_COUNT_TOKENS",
        "LANGSLICE_GENAI_AP_USE_FILE_API",
        "LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE",
        "LANGSLICE_GENAI_AP_USE_INTERACTIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    closed: list[bool] = []
    monkeypatch.setattr(settings_dialog, "close_client", lambda: closed.append(True))

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "LANGSLICE_GENAI_BACKEND=ai_studio",
                "LANGSLICE_GENAI_COUNT_TOKENS=1",
                "LANGSLICE_GENAI_AP_USE_FILE_API=0",
                "LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE=1",
                "LANGSLICE_GENAI_AP_USE_INTERACTIONS=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    app = QApplication.instance() or QApplication([])
    dialog = settings_dialog.SettingsDialog()
    dialog.env_path = str(env_path)
    dialog._load_current_settings()

    assert dialog.count_tokens_checkbox.isChecked() is True
    assert dialog.ap_use_file_api_checkbox.isChecked() is False
    assert dialog.ap_use_context_cache_checkbox.isChecked() is True
    assert dialog.ap_use_interactions_checkbox.isChecked() is False

    dialog.count_tokens_checkbox.setChecked(False)
    dialog.ap_use_file_api_checkbox.setChecked(True)
    dialog.ap_use_context_cache_checkbox.setChecked(False)
    dialog.ap_use_interactions_checkbox.setChecked(True)
    dialog._save_settings()

    saved = dotenv.dotenv_values(env_path)
    assert saved["LANGSLICE_GENAI_COUNT_TOKENS"] == "0"
    assert saved["LANGSLICE_GENAI_AP_USE_FILE_API"] == "1"
    assert saved["LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE"] == "0"
    assert saved["LANGSLICE_GENAI_AP_USE_INTERACTIONS"] == "1"
    assert closed == [True]
    assert app is not None
    dialog.close()
