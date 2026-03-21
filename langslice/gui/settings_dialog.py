"""Settings dialog for configuring VLM providers."""

import os
from typing import Any

import dotenv

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from langslice.gui.theme import TEXT_SECONDARY
from langslice.ai.config import (
    _BACKEND_AI_STUDIO,
    _BACKEND_VERTEX_ADC,
    _BACKEND_VERTEX_API_KEY,
    close_client,
)

_ENV_COUNT_TOKENS = "LANGSLICE_GENAI_COUNT_TOKENS"
_ENV_AP_USE_FILE_API = "LANGSLICE_GENAI_AP_USE_FILE_API"
_ENV_AP_USE_CONTEXT_CACHE = "LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE"
_ENV_AP_USE_INTERACTIONS = "LANGSLICE_GENAI_AP_USE_INTERACTIONS"


class SettingsDialog(QDialog):
    """Dialog to configure VLM provider and keys."""

    settings_saved = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        # Determine current .env file path
        # Assuming we are running from the project root C:\LabSoftware\LangSlice
        self.env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

        self._build_ui()
        self._load_current_settings()
        self._on_provider_changed()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Main Form
        self.form = QFormLayout()
        self.form.setSpacing(12)

        # Provider Selection
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Google AI Studio", _BACKEND_AI_STUDIO)
        self.provider_combo.addItem("Vertex AI (ADC)", _BACKEND_VERTEX_ADC)
        self.provider_combo.addItem("Vertex AI (API Key)", _BACKEND_VERTEX_API_KEY)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.form.addRow("VLM Provider:", self.provider_combo)

        # AI Studio API Key
        self.ai_studio_key = QLineEdit()
        self.ai_studio_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_studio_key.setPlaceholderText("AIzaSy...")
        self.form.addRow("AI Studio API Key:", self.ai_studio_key)

        # Vertex Project ID
        self.vertex_project = QLineEdit()
        self.vertex_project.setPlaceholderText("my-gcp-project-123")
        self.form.addRow("GCP Project ID:", self.vertex_project)

        # Vertex Location
        self.vertex_location = QLineEdit()
        self.vertex_location.setPlaceholderText("us-central1")
        self.form.addRow("GCP Location:", self.vertex_location)

        # Vertex API Key
        self.vertex_key = QLineEdit()
        self.vertex_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.vertex_key.setPlaceholderText("AIzaSy...")
        self.form.addRow("Vertex API Key:", self.vertex_key)

        layout.addLayout(self.form)

        advanced_wrap = QWidget()
        advanced_layout = QVBoxLayout(advanced_wrap)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(8)

        advanced_title = QLabel("Advanced AP Transport")
        advanced_title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 600;")
        advanced_layout.addWidget(advanced_title)

        self.count_tokens_checkbox = QCheckBox("Enable token counting preflight")
        self.count_tokens_checkbox.setToolTip(
            "Runs an extra Gemini count_tokens call before AP estimation and registration single-pass requests."
        )
        advanced_layout.addWidget(self.count_tokens_checkbox)

        self.ap_use_file_api_checkbox = QCheckBox("Use File API for AP estimation")
        self.ap_use_file_api_checkbox.setToolTip(
            "Uploads the AP slice through Gemini File API instead of sending it inline."
        )
        advanced_layout.addWidget(self.ap_use_file_api_checkbox)

        self.ap_use_context_cache_checkbox = QCheckBox("Use context cache for AP estimation")
        self.ap_use_context_cache_checkbox.setToolTip(
            "Caches the AP prompt context when supported to reduce repeated prompt cost."
        )
        advanced_layout.addWidget(self.ap_use_context_cache_checkbox)

        self.ap_use_interactions_checkbox = QCheckBox("Use Interactions API for AP estimation")
        self.ap_use_interactions_checkbox.setToolTip(
            "Uses Gemini Interactions API for AP estimation when supported by the active backend."
        )
        advanced_layout.addWidget(self.ap_use_interactions_checkbox)

        layout.addWidget(advanced_wrap)

        # Info Box
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.info_label)

        # Buttons
        button_box = QHBoxLayout()
        button_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondary")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save_settings)

        button_box.addWidget(self.btn_cancel)
        button_box.addWidget(self.btn_save)
        layout.addLayout(button_box)

    def _load_current_settings(self) -> None:
        """Load values directly from the .env file to ensure we get the saved keys."""
        env_vars = {}
        if os.path.exists(self.env_path):
            env_vars = dotenv.dotenv_values(self.env_path)

        # Fall back to os.environ if not in .env (e.g., set externally in terminal)
        def get_val(key: str, default: str = "") -> str:
            val = env_vars.get(key)
            if val is None:
                val = os.environ.get(key, default)
            return val

        def get_bool(key: str) -> bool:
            return get_val(key, "").strip().lower() in {"1", "true", "yes", "on"}

        backend = get_val("LANGSLICE_GENAI_BACKEND", _BACKEND_AI_STUDIO).lower()
        if backend == _BACKEND_AI_STUDIO:
            self.provider_combo.setCurrentIndex(0)
        elif backend == _BACKEND_VERTEX_ADC:
            self.provider_combo.setCurrentIndex(1)
        elif backend == _BACKEND_VERTEX_API_KEY:
            self.provider_combo.setCurrentIndex(2)

        self.ai_studio_key.setText(get_val("GEMINI_API_KEY", ""))
        self.vertex_project.setText(get_val("GOOGLE_CLOUD_PROJECT", ""))
        self.vertex_location.setText(get_val("GOOGLE_CLOUD_LOCATION", "us-central1"))

        # Vertex API Key might be under VERTEX_API_KEY in older configs
        v_key = get_val("GOOGLE_CLOUD_API_KEY", get_val("VERTEX_API_KEY", ""))
        self.vertex_key.setText(v_key)
        self.count_tokens_checkbox.setChecked(get_bool(_ENV_COUNT_TOKENS))
        self.ap_use_file_api_checkbox.setChecked(get_bool(_ENV_AP_USE_FILE_API))
        self.ap_use_context_cache_checkbox.setChecked(get_bool(_ENV_AP_USE_CONTEXT_CACHE))
        self.ap_use_interactions_checkbox.setChecked(get_bool(_ENV_AP_USE_INTERACTIONS))

    def _on_provider_changed(self) -> None:
        """Show/hide fields based on the selected provider."""
        provider = self.provider_combo.currentData()

        if provider == _BACKEND_AI_STUDIO:
            self._set_row_visible(self.ai_studio_key, True)
            self._set_row_visible(self.vertex_project, False)
            self._set_row_visible(self.vertex_location, False)
            self._set_row_visible(self.vertex_key, False)
            self.info_label.setText(
                "Uses the Google AI Studio free tier. Requires a Gemini API Key."
            )

        elif provider == _BACKEND_VERTEX_ADC:
            self._set_row_visible(self.ai_studio_key, False)
            self._set_row_visible(self.vertex_project, True)
            self._set_row_visible(self.vertex_location, True)
            self._set_row_visible(self.vertex_key, False)
            self.info_label.setText(
                "Uses Vertex AI with Application Default Credentials (ADC). "
                "Ensure you have run `gcloud auth application-default login`."
            )

        elif provider == _BACKEND_VERTEX_API_KEY:
            self._set_row_visible(self.ai_studio_key, False)
            self._set_row_visible(self.vertex_project, True)
            self._set_row_visible(self.vertex_location, True)
            self._set_row_visible(self.vertex_key, True)
            self.info_label.setText("Uses Vertex AI with a specific API Key tied to a GCP Project.")

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        """Hide both the widget and its associated label in the FormLayout."""
        label = self.form.labelForField(widget)
        if label:
            label.setVisible(visible)
        widget.setVisible(visible)

    def _save_bool_setting(self, key: str, enabled: bool) -> None:
        value = "1" if enabled else "0"
        os.environ[key] = value
        dotenv.set_key(self.env_path, key, value)

    def _save_settings(self) -> None:
        """Save settings to os.environ and to the .env file."""
        provider = self.provider_combo.currentData()

        # Update os.environ so it takes effect immediately in the current process
        os.environ["LANGSLICE_GENAI_BACKEND"] = provider

        # We write to dotenv so it persists across restarts
        if not os.path.exists(self.env_path):
            open(self.env_path, "a").close()

        dotenv.set_key(self.env_path, "LANGSLICE_GENAI_BACKEND", provider)

        if provider == _BACKEND_AI_STUDIO:
            val = self.ai_studio_key.text().strip()
            os.environ["GEMINI_API_KEY"] = val
            dotenv.set_key(self.env_path, "GEMINI_API_KEY", val)

        elif provider == _BACKEND_VERTEX_ADC:
            proj = self.vertex_project.text().strip()
            loc = self.vertex_location.text().strip()
            os.environ["GOOGLE_CLOUD_PROJECT"] = proj
            os.environ["GOOGLE_CLOUD_LOCATION"] = loc
            dotenv.set_key(self.env_path, "GOOGLE_CLOUD_PROJECT", proj)
            dotenv.set_key(self.env_path, "GOOGLE_CLOUD_LOCATION", loc)

        elif provider == _BACKEND_VERTEX_API_KEY:
            proj = self.vertex_project.text().strip()
            loc = self.vertex_location.text().strip()
            key = self.vertex_key.text().strip()
            os.environ["GOOGLE_CLOUD_PROJECT"] = proj
            os.environ["GOOGLE_CLOUD_LOCATION"] = loc
            os.environ["GOOGLE_CLOUD_API_KEY"] = key
            dotenv.set_key(self.env_path, "GOOGLE_CLOUD_PROJECT", proj)
            dotenv.set_key(self.env_path, "GOOGLE_CLOUD_LOCATION", loc)
            dotenv.set_key(self.env_path, "GOOGLE_CLOUD_API_KEY", key)

        self._save_bool_setting(_ENV_COUNT_TOKENS, self.count_tokens_checkbox.isChecked())
        self._save_bool_setting(_ENV_AP_USE_FILE_API, self.ap_use_file_api_checkbox.isChecked())
        self._save_bool_setting(
            _ENV_AP_USE_CONTEXT_CACHE,
            self.ap_use_context_cache_checkbox.isChecked(),
        )
        self._save_bool_setting(
            _ENV_AP_USE_INTERACTIONS,
            self.ap_use_interactions_checkbox.isChecked(),
        )

        close_client()

        self.settings_saved.emit()
        self.accept()
