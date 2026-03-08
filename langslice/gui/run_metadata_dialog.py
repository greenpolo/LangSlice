"""Dialog for classifying a debug trace with optional ground-truth AP."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from langslice.gui.theme import ERROR, SUCCESS, TEXT_SECONDARY


class RunMetadataDialog(QDialog):
    """Collect optional evaluation metadata for a classified debug trace."""

    def __init__(
        self,
        *,
        outcome: str,
        estimated_position_mm: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._outcome = outcome
        self._estimated_position_mm = float(estimated_position_mm)

        self.setWindowTitle(f"Mark {outcome.title()}")
        self.setMinimumWidth(420)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        intro = QLabel(
            "Save optional run metadata for later analysis. "
            "Ground-truth AP lets LangSlice compute estimation error automatically."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(12)

        outcome_color = SUCCESS if self._outcome == "success" else ERROR
        self.outcome_value = QLabel(self._outcome.title())
        self.outcome_value.setStyleSheet(f"font-weight: 600; color: {outcome_color};")
        form.addRow("Outcome:", self.outcome_value)

        self.estimated_value = QLabel(f"{self._estimated_position_mm:.3f} mm")
        form.addRow("Estimated AP:", self.estimated_value)

        self.has_ground_truth = QCheckBox("Record ground-truth AP")
        self.has_ground_truth.toggled.connect(self._on_ground_truth_toggled)
        form.addRow("Ground Truth:", self.has_ground_truth)

        self.actual_position_spin = QDoubleSpinBox()
        self.actual_position_spin.setRange(-999.0, 999.0)
        self.actual_position_spin.setDecimals(3)
        self.actual_position_spin.setSingleStep(0.05)
        self.actual_position_spin.setSuffix(" mm")
        self.actual_position_spin.setEnabled(False)
        form.addRow("Actual AP:", self.actual_position_spin)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Optional notes about landmarks, failure mode, or why this run was marked this way."
        )
        self.notes_edit.setMinimumHeight(110)
        form.addRow("Notes:", self.notes_edit)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("secondary")
        cancel_button.clicked.connect(self.reject)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)

        button_row.addWidget(cancel_button)
        button_row.addWidget(save_button)
        layout.addLayout(button_row)

    def _on_ground_truth_toggled(self, checked: bool) -> None:
        self.actual_position_spin.setEnabled(checked)
        if checked:
            self.actual_position_spin.setFocus(Qt.FocusReason.TabFocusReason)

    def metadata(self) -> dict[str, Any]:
        actual_position_mm: float | None = None
        if self.has_ground_truth.isChecked():
            actual_position_mm = float(self.actual_position_spin.value())

        return {
            "outcome": self._outcome,
            "estimated_position_mm": self._estimated_position_mm,
            "actual_position_mm": actual_position_mm,
            "notes": self.notes_edit.toPlainText().strip(),
        }
