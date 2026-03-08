"""Dark theme stylesheet for LangSlice PySide6 GUI.
Ported from src/index.css with a #0a0a0a background, #6366f1 indigo accent,
and glass panels with translucent backgrounds and subtle borders.
"""

# ---------------------------------------------------------------------------
# Colour constants (referenced by widgets that need programmatic colours)
# ---------------------------------------------------------------------------
BG_PRIMARY = "#0a0a0a"
BG_PANEL = "rgba(20, 20, 20, 153)"  # 0.6 opacity
BG_PANEL_SOLID = "#141414"  # fallback when translucency isn't supported
BORDER_SUBTLE = "rgba(255, 255, 255, 20)"  # 0.08 opacity
TEXT_PRIMARY = "#ededed"
TEXT_SECONDARY = "#888888"
TEXT_WHITE = "#ffffff"
ACCENT = "#6366f1"
ACCENT_HOVER = "#818cf8"
ACCENT_PRESSED = "#4f46e5"
SUCCESS = "#22c55e"
WARNING = "#eab308"
ERROR = "#ef4444"

# Font families - Inter for UI, JetBrains Mono for data
FONT_SANS = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
FONT_MONO = '"JetBrains Mono", "Cascadia Code", "Consolas", monospace'

# ---------------------------------------------------------------------------
# Main application stylesheet (Qt CSS)
# ---------------------------------------------------------------------------
STYLESHEET = """
/* ── Global ─────────────────────────────────────────────────── */
QWidget {
    background-color: #0a0a0a;
    color: #ededed;
    font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* ── Glass Panels ───────────────────────────────────────────── */
QFrame#glassPanel, QGroupBox#glassPanel {
    background-color: #141414;
    border: 1px solid rgba(255, 255, 255, 20);
    border-radius: 8px;
}

/* ── Labels ─────────────────────────────────────────────────── */
QLabel {
    background: transparent;
    border: none;
}

QLabel#heading {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
}

QLabel#subheading {
    font-size: 12px;
    color: #888888;
}

QLabel#monoLabel {
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    color: #888888;
    text-transform: uppercase;
}

QLabel#monoValue {
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 14px;
    color: #ffffff;
}

/* ── Buttons ────────────────────────────────────────────────── */
QPushButton {
    background-color: #6366f1;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #818cf8;
}
QPushButton:pressed {
    background-color: #4f46e5;
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555555;
}
QPushButton#secondary {
    background-color: transparent;
    border: 1px solid rgba(255, 255, 255, 20);
    color: #ededed;
}
QPushButton#secondary:hover {
    background-color: rgba(255, 255, 255, 10);
}

/* ── ComboBox ───────────────────────────────────────────────── */
QComboBox {
    background-color: #141414;
    border: 1px solid rgba(255, 255, 255, 20);
    border-radius: 6px;
    padding: 6px 12px;
    color: #ededed;
    font-size: 13px;
    min-width: 180px;
}
QComboBox:hover {
    border-color: rgba(255, 255, 255, 40);
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #888888;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1a1a1a;
    border: 1px solid rgba(255, 255, 255, 20);
    color: #ededed;
    selection-background-color: #6366f1;
    selection-color: #ffffff;
    outline: none;
}

/* ── Slider ─────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    background: #2a2a2a;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #6366f1;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #818cf8;
}
QSlider::sub-page:horizontal {
    background: #6366f1;
    border-radius: 2px;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
QScrollBar:vertical {
    background: rgba(0, 0, 0, 50);
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 25);
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 50);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: rgba(0, 0, 0, 50);
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 25);
    border-radius: 4px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: rgba(255, 255, 255, 50);
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Text Edits / Log Panel ─────────────────────────────────── */
QTextEdit, QPlainTextEdit {
    background-color: #111111;
    border: 1px solid rgba(255, 255, 255, 20);
    border-radius: 6px;
    color: #ededed;
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: #6366f1;
}
QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #6366f1;
}

/* ── Tab Widget (view modes) ────────────────────────────────── */
QTabBar::tab {
    background: transparent;
    color: #888888;
    padding: 8px 16px;
    border: none;
    font-size: 12px;
    font-weight: 500;
}
QTabBar::tab:selected {
    color: #ffffff;
    border-bottom: 2px solid #6366f1;
}
QTabBar::tab:hover:!selected {
    color: #bbbbbb;
}
QTabWidget::pane {
    border: none;
    background: transparent;
}

/* ── Progress Bar ───────────────────────────────────────────── */
QProgressBar {
    background-color: #2a2a2a;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background-color: #6366f1;
    border-radius: 3px;
}

/* ── Step Indicator (custom object names) ───────────────────── */
QLabel#stepComplete {
    color: #22c55e;
    font-weight: 600;
}
QLabel#stepActive {
    color: #6366f1;
    font-weight: 600;
}
QLabel#stepPending {
    color: #555555;
}

/* ── Tooltip ────────────────────────────────────────────────── */
QToolTip {
    background-color: #1a1a1a;
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 4px;
    color: #ededed;
    padding: 4px 8px;
    font-size: 12px;
}
"""
