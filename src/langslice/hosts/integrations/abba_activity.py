"""Live companion window: agent-visible images beside a streaming activity log.

The host pushes sanitized, in-memory events. Swing drains the queue on its own
thread; no UI object or image bytes enter the agent's context or JSONL trace.
"""

from __future__ import annotations

import html
import io
import queue
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BATCHES = 40
MAX_LOG_CHARS = 180_000


@dataclass
class ImageBatch:
    label: str
    images: list[dict[str, Any]]


@dataclass
class ActivityHistory:
    """Bounded image history; each encoded image is the agent's actual input."""

    batches: deque[ImageBatch] = field(default_factory=deque)
    byte_count: int = 0
    removed: int = 0

    def add(self, label: str, images: list[dict[str, Any]]) -> None:
        clean = []
        for image in images:
            data = image.get("data")
            if not isinstance(data, bytes) or not data:
                continue
            if len(data) > MAX_IMAGE_BYTES:
                continue
            clean.append(
                {
                    "data": data,
                    "mime_type": str(image.get("mime_type", "image/png")),
                    "label": str(image.get("label") or "Image"),
                }
            )
        if not clean:
            return
        # A single giant batch is bounded too, without keeping hidden copies.
        total = sum(len(image["data"]) for image in clean)
        start = 0
        while total > MAX_IMAGE_BYTES:
            total -= len(clean[start]["data"])
            start += 1
            self.removed += 1
        clean = clean[start:]
        self.batches.append(ImageBatch(label, clean))
        self.byte_count += sum(len(image["data"]) for image in clean)
        while len(self.batches) > MAX_IMAGE_BATCHES or self.byte_count > MAX_IMAGE_BYTES:
            old = self.batches.popleft()
            self.byte_count -= sum(len(image["data"]) for image in old.images)
            self.removed += 1


def _tool_title(name: str) -> str:
    return name.replace("_", " ").strip() or "tool"


def format_event(event: dict[str, Any]) -> tuple[str, str]:
    """Readable public activity; protocol payloads belong in the disk trace."""
    kind = event.get("kind", "")
    text = event.get("text", "")
    if kind in {"text", "reasoning"}:
        prefix = "Updated summary\n" if event.get("revised") else ""
        return prefix + str(text), kind
    if kind == "tool_start":
        targets = event.get("target_ids") or []
        labels = []
        for target in targets[:4]:
            match = re.fullmatch(r"section_0*(\d+)\.[^.]+", str(target))
            labels.append(f"slice {match[1]}" if match else str(target)[:48])
        suffix = " · " + ", ".join(labels) if labels else ""
        if len(targets) > 4:
            suffix += f" + {len(targets) - 4} more"
        return f"Called {_tool_title(str(event.get('name', '')))}{suffix}", "tool"
    if kind in {"seed", "tool_result"}:
        count = len(event.get("images") or [])
        response = event.get("response") or {}
        if isinstance(response, dict):
            status = str(response.get("status", ""))
            if status and status.lower() not in {"ok", "success"}:
                reason = response.get("message") or response.get("reason") or status
                title = _tool_title(str(event.get("name", "Tool")))
                return f"{title}: {str(reason)[:500]}", "error"
        if count:
            return f"Viewing {count} image{'s' if count != 1 else ''}", "result"
        return "", "muted"
    if kind == "complete":
        return "Submitted" if event.get("submitted") else "Stopped before submission", "tool"
    if kind == "error":
        return f"Run failed: {text}", "error"
    # Usage, seed instructions, paths and JSON make the conversation harder
    # to read. They remain available in console/opt-in JSONL diagnostics.
    return "", "muted"


def _rich_text(text: str) -> str:
    """Small, escaped Markdown subset suitable for Swing's HTML renderer."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped, flags=re.S)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"(?m)^#{1,4} (.+)$", r"<b>\1</b>", escaped)
    return escaped.replace("\n", "<br>")


def transcript_html(entries: list[tuple[str, str]]) -> str:
    blocks = []
    for text, style in entries:
        body = _rich_text(text)
        if style in {"tool", "result"}:
            color = "#74d9cb" if style == "tool" else "#9cb4c4"
            blocks.append(
                f'<p style="margin:6px 0;color:{color};font-size:12px">&#8250; {body}</p>'
            )
        elif style == "error":
            blocks.append(f'<p style="color:#ffaaa2">{body}</p>')
        else:
            label = "Reasoning summary" if style == "reasoning" else "LangSlice"
            if style == "muted":
                label = ""
            blocks.append(
                '<p style="margin-top:18px;margin-bottom:4px;color:#8399ab;'
                f'font-size:10px">{label}</p>'
                f'<p style="margin:0 0 14px;color:#e2eaf0">{body}</p>'
            )
    return (
        "<html><head><style>body {font-family:SansSerif;font-size:14px;"
        "background:#121e28;margin:14px;} p {margin:8px 0;} "
        "code {font-family:Monospaced;}</style></head><body>" + "".join(blocks) + "</body></html>"
    )


class ActivityWindow:
    """A vertical companion beside ABBA, safe to feed from a worker thread."""

    def __init__(self, *, parent: Any = None) -> None:
        from jpype import JProxy  # pyright: ignore[reportMissingImports]
        from scyjava import jimport  # pyright: ignore[reportMissingImports]

        self._j = jimport
        self._proxy = JProxy
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)
        self._queue_lock = threading.Lock()
        self._queued_bytes = 0
        self._decoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langslice-images")
        self._render_generation = 0
        self._render_future: Any = None
        self._history = ActivityHistory()
        self._index = -1
        self._page = 0
        self._refs: list[Any] = []
        self.closed = False
        self._dropped = 0
        self._last_style = ""
        self._entries: list[tuple[str, str]] = []
        self._log_dirty = False
        self._dirty_images = True
        self._last_size = (0, 0)
        self._colors = {
            "bg": (12, 20, 28),
            "panel": (18, 30, 40),
            "text": (225, 235, 241),
            "muted": (137, 159, 176),
            "tool": (85, 220, 198),
            "reasoning": (193, 177, 245),
            "result": (175, 197, 209),
            "error": (255, 150, 145),
        }
        self._edt(lambda: self._build(parent), wait=True)

    def _edt(self, fn: Any, *, wait: bool = False) -> None:
        swing = self._j("javax.swing.SwingUtilities")
        if swing.isEventDispatchThread():
            fn()
            return
        runnable = self._proxy("java.lang.Runnable", dict(run=fn))
        if wait:
            swing.invokeAndWait(runnable)
        else:
            swing.invokeLater(runnable)

    def _color(self, name: str) -> Any:
        return self._j("java.awt.Color")(*self._colors[name])

    def _label(self, text: str, *, size: int = 14, style: str = "text") -> Any:
        label = self._j("javax.swing.JLabel")(text)
        label.setForeground(self._color(style))
        label.setFont(self._j("java.awt.Font")("SansSerif", 0, size))
        return label

    def _button(self, label: str, fn: Any) -> Any:
        button = self._j("javax.swing.JButton")(label)
        callback = self._proxy(
            "java.awt.event.ActionListener", dict(actionPerformed=lambda e: fn())
        )
        self._refs.append(callback)
        button.addActionListener(callback)
        button.setFocusable(False)
        return button

    def _build(self, parent: Any) -> None:
        j = self._j
        Border = j("java.awt.BorderLayout")
        Panel = j("javax.swing.JPanel")
        EmptyBorder = j("javax.swing.BorderFactory").createEmptyBorder
        self.frame = j("javax.swing.JFrame")("LangSlice · Agent activity")
        self.frame.setDefaultCloseOperation(j("javax.swing.WindowConstants").HIDE_ON_CLOSE)
        self.frame.setSize(560, 960)
        self.frame.setMinimumSize(j("java.awt.Dimension")(380, 620))
        if parent is not None:
            config = parent.getGraphicsConfiguration()
            bounds = config.getBounds()
            insets = j("java.awt.Toolkit").getDefaultToolkit().getScreenInsets(config)
            x, y = bounds.x + insets.left, bounds.y + insets.top
            width = bounds.width - insets.left - insets.right
            height = bounds.height - insets.top - insets.bottom
            side = min(620, max(380, round(width * 0.30)))
            parent.setExtendedState(j("java.awt.Frame").NORMAL)
            parent.setBounds(x, y, width - side, height)
            self.frame.setBounds(x + width - side, y, side, height)
        else:
            self.frame.setLocationRelativeTo(None)
        root = Panel(Border(0, 12))
        root.setBackground(self._color("bg"))
        root.setBorder(EmptyBorder(14, 14, 10, 14))
        header = Panel(Border())
        header.setOpaque(False)
        header.add(self._label("LangSlice", size=22), Border.NORTH)
        self.status = self._label("Preparing sections…", style="tool")
        header.add(self.status, Border.SOUTH)
        root.add(header, Border.NORTH)

        left = Panel(Border(0, 10))
        left.setBackground(self._color("bg"))
        image_header = Panel(Border())
        image_header.setOpaque(False)
        self.image_title = self._label("What the agent sees", size=17)
        image_header.add(self.image_title, Border.NORTH)
        self.image_detail = self._label(
            "Images will appear when the session is prepared", style="muted"
        )
        image_header.add(self.image_detail, Border.SOUTH)
        left.add(image_header, Border.NORTH)
        self.image_grid = Panel(j("java.awt.GridLayout")(1, 1, 10, 10))
        self.image_grid.setBackground(self._color("bg"))
        self.image_grid.add(
            self._label("Preparing the agent's initial view…", size=20, style="muted")
        )
        left.add(self.image_grid, Border.CENTER)
        controls = Panel(j("java.awt.FlowLayout")(j("java.awt.FlowLayout").LEFT, 6, 0))
        controls.setOpaque(False)
        self.previous = self._button("←", lambda: self._browse(-1))
        self.next = self._button("→", lambda: self._browse(1))
        self.latest = self._button("Latest", self._follow_latest)
        self.page_button = self._button("More", self._next_page)
        self.follow = j("javax.swing.JCheckBox")("Follow images", True)
        self.follow.setOpaque(False)
        self.follow.setForeground(self._color("text"))
        for button in (self.previous, self.next, self.latest, self.page_button, self.follow):
            controls.add(button)
        left.add(controls, Border.SOUTH)

        right = Panel(Border(0, 8))
        right.setBackground(self._color("bg"))
        right_header = Panel(Border())
        right_header.setOpaque(False)
        right_header.add(self._label("Agent activity", size=16), Border.NORTH)
        right.add(right_header, Border.NORTH)
        self.log = j("javax.swing.JTextPane")()
        self.log.setEditable(False)
        self.log.setBackground(self._color("panel"))
        self.log.setForeground(self._color("text"))
        self.log.setContentType("text/html")
        self.log.setText(transcript_html([]))
        self.log_scroll = j("javax.swing.JScrollPane")(self.log)
        self.log_scroll.setBorder(EmptyBorder(0, 0, 0, 0))
        right.add(self.log_scroll, Border.CENTER)
        self.autoscroll = j("javax.swing.JCheckBox")("Follow activity", True)
        self.autoscroll.setOpaque(False)
        self.autoscroll.setForeground(self._color("text"))
        right.add(self.autoscroll, Border.SOUTH)

        split = j("javax.swing.JSplitPane")(j("javax.swing.JSplitPane").VERTICAL_SPLIT, right, left)
        split.setBorder(EmptyBorder(0, 0, 0, 0))
        split.setResizeWeight(0.62)
        split.setDividerSize(10)
        root.add(split, Border.CENTER)
        footer = self._label("Live in ABBA • Reasoning summaries when available", style="muted")
        root.add(footer, Border.SOUTH)
        self.footer = footer
        self.frame.setContentPane(root)
        self.frame.setVisible(True)
        split.setDividerLocation(0.62)
        self.split = split
        callback = self._proxy(
            "java.awt.event.ActionListener", dict(actionPerformed=lambda e: self._drain())
        )
        self._refs.append(callback)
        self._timer = j("javax.swing.Timer")(80, callback)
        self._timer.start()

    def on_event(self, event: dict[str, Any]) -> None:
        """Nonblocking: a slow/hidden viewer can never hold up registration."""
        if self.closed:
            return
        size = sum(len(im.get("data", b"")) for im in event.get("images", []))
        size += len(event.get("text", "")) * 4
        with self._queue_lock:
            if self._queued_bytes + size > MAX_IMAGE_BYTES:
                self._dropped += 1
                return
            try:
                self._queue.put_nowait({**event, "_viewer_bytes": size})
                self._queued_bytes += size
            except queue.Full:
                self._dropped += 1

    def show(self) -> None:
        if self.closed:
            return

        def show() -> None:
            self.frame.setVisible(True)
            self.frame.toFront()

        self._edt(show)

    def dispose(self) -> None:
        self.closed = True
        self._decoder.shutdown(wait=False, cancel_futures=True)

        def close() -> None:
            self._timer.stop()
            self.frame.dispose()
            self._history.batches.clear()
            self._history.byte_count = 0
            self.image_grid.removeAll()
            self.log.setText("")
            self._entries.clear()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queued_bytes = 0

        self._edt(close)

    def _append(self, text: str, style: str) -> None:
        if not text:
            return
        if self._entries and style in {"reasoning", "text"} and self._last_style == style:
            prior, _ = self._entries[-1]
            self._entries[-1] = (prior + text, style)
        else:
            self._entries.append((text, style))
        self._last_style = style
        total = sum(len(entry[0]) for entry in self._entries)
        removed = False
        while total > MAX_LOG_CHARS and len(self._entries) > 1:
            total -= len(self._entries.pop(0)[0])
            removed = True
        if total > MAX_LOG_CHARS:
            content, entry_style = self._entries[0]
            self._entries[0] = (content[-MAX_LOG_CHARS:], entry_style)
            removed = True
        if removed:
            self.footer.setText("Earlier activity removed • ABBA updates continue")
        self._log_dirty = True

    def _refresh_log(self) -> None:
        if not self._log_dirty:
            return
        scrollbar = self.log_scroll.getVerticalScrollBar()
        position = scrollbar.getValue()
        self.log.setText(transcript_html(self._entries))
        if self.autoscroll.isSelected():
            self.log.setCaretPosition(self.log.getDocument().getLength())
        else:
            self.log.setCaretPosition(0)
            self._edt(lambda: scrollbar.setValue(position))
        self._log_dirty = False

    def _drain(self) -> None:
        try:
            if self._dropped:
                self._append(f"\n[Viewer skipped {self._dropped} updates while busy]\n", "muted")
                self._dropped = 0
            for _ in range(120):
                try:
                    with self._queue_lock:
                        event = self._queue.get_nowait()
                        self._queued_bytes -= event.get("_viewer_bytes", 0)
                except queue.Empty:
                    break
                kind = event.get("kind")
                if kind == "status":
                    self.status.setText(str(event.get("text", "")))
                    continue
                self._append(*format_event(event))
                if kind == "tool_start":
                    self.status.setText(f"Running {_tool_title(str(event.get('name', 'tool')))}…")
                elif kind in {"text", "reasoning", "session", "tool_result"}:
                    self.status.setText("Agent exploring…")
                elif kind == "complete":
                    self.status.setText("Submitted" if event.get("submitted") else "Stopped")
                elif kind == "error":
                    self.status.setText("Run failed")
                images = event.get("images")
                if images:
                    label = (
                        "Initial view" if kind == "seed" else str(event.get("name") or "Agent view")
                    )
                    selected = (
                        self._history.batches[self._index]
                        if 0 <= self._index < len(self._history.batches)
                        else None
                    )
                    self._history.add(label, images)
                    if selected is not None:
                        self._index = next(
                            (
                                i
                                for i, batch in enumerate(self._history.batches)
                                if batch is selected
                            ),
                            0,
                        )
                    self._dirty_images = True
                    if self.follow.isSelected() or self._index < 0:
                        self._index = len(self._history.batches) - 1
                        self._page = 0
                        self._dirty_images = True
            self._refresh_log()
            size = (self.image_grid.getWidth(), self.image_grid.getHeight())
            if size != self._last_size:
                self._dirty_images = True
                self._last_size = size
            if self._dirty_images and self._history.batches:
                self._render_images()
                self._dirty_images = False
        except Exception as exc:
            # Display failures are visible but remain outside the agent path.
            self.status.setText("Viewer update failed")
            self._append(f"\nViewer error: {exc}\n", "error")
            self._refresh_log()

    def _browse(self, direction: int) -> None:
        self.follow.setSelected(False)
        self._index = min(max(0, self._index + direction), len(self._history.batches) - 1)
        self._page = 0
        self._dirty_images = True

    def _follow_latest(self) -> None:
        self.follow.setSelected(True)
        self._index = len(self._history.batches) - 1
        self._page = 0
        self._dirty_images = True

    def _next_page(self) -> None:
        if self._history.batches:
            batch = self._history.batches[self._index]
            self._page = (self._page + 1) % max(1, (len(batch.images) + 3) // 4)
            self._dirty_images = True

    def _render_images(self) -> None:
        self._index = min(max(0, self._index), len(self._history.batches) - 1)
        batch = self._history.batches[self._index]
        self._page = min(self._page, max(0, (len(batch.images) - 1) // 4))
        images = batch.images[self._page * 4 : (self._page + 1) * 4]
        grid_width = max(160, self.image_grid.getWidth() - 16)
        grid_height = max(120, self.image_grid.getHeight() - 16)
        self._render_generation += 1
        generation = self._render_generation
        page = self._page
        self.image_title.setText(batch.label)
        first = self._page * 4 + 1
        last = min(len(batch.images), first + 3)
        self.image_detail.setText(
            f"Images {first}–{last} of {len(batch.images)}  ·  "
            f"View {self._index + 1} of {len(self._history.batches)}"
        )
        self.previous.setEnabled(self._index > 0)
        self.next.setEnabled(self._index < len(self._history.batches) - 1)
        self.page_button.setEnabled(len(batch.images) > 4)
        if self._history.removed:
            self.footer.setText(
                "Older images removed to bound viewer memory • ABBA updates continue"
            )
        if self._render_future is not None:
            self._render_future.cancel()

        def prepare() -> None:
            from PIL import Image

            dimensions = []
            for image in images:
                try:
                    with Image.open(io.BytesIO(image["data"])) as im:
                        dimensions.append(im.size)
                except Exception:
                    dimensions.append((1, 1))

            # Wide montages need stacked rows; individual sections benefit
            # from a two-column grid. Choose the fit with most visible pixels.
            def layout_area(columns: int) -> float:
                rows = (len(images) + columns - 1) // columns
                width = max(80, grid_width // columns - 12)
                height = max(60, grid_height // rows - 42)
                return sum(w * h * min(width / w, height / h) ** 2 for w, h in dimensions)

            columns = max(range(1, min(2, len(images)) + 1), key=layout_area)
            rows = (len(images) + columns - 1) // columns
            width = max(80, grid_width // columns - 12)
            height = max(60, grid_height // rows - 42)
            decoded = []
            for image in images:
                try:
                    with Image.open(io.BytesIO(image["data"])) as im:
                        factor = min(width / im.width, height / im.height)
                        resized = im.convert("RGB").resize(
                            (max(1, int(im.width * factor)), max(1, int(im.height * factor))),
                            Image.Resampling.LANCZOS,
                        )
                        output = io.BytesIO()
                        resized.save(output, format="PNG")
                        decoded.append({"data": output.getvalue(), "label": image["label"]})
                except Exception:
                    decoded.append({"data": b"", "label": "Image could not be decoded"})
            if not self.closed:
                self._edt(lambda: self._publish_images(decoded, rows, columns, page, generation))

        self._render_future = self._decoder.submit(prepare)

    def _publish_images(
        self, images: list[dict[str, Any]], rows: int, columns: int, page: int, generation: int
    ) -> None:
        from jpype import JArray, JByte  # pyright: ignore[reportMissingImports]

        if self.closed or generation != self._render_generation:
            return
        j = self._j
        self.image_grid.removeAll()
        self.image_grid.setLayout(j("java.awt.GridLayout")(max(1, rows), columns, 10, 10))
        for index, image in enumerate(images):
            panel = j("javax.swing.JPanel")(j("java.awt.BorderLayout")())
            panel.setBackground(self._color("panel"))
            if image["data"]:
                icon = j("javax.swing.ImageIcon")(JArray(JByte)(image["data"]))
                label = j("javax.swing.JLabel")(icon)
            else:
                label = self._label("Image could not be decoded", style="error")
            panel.add(label, j("java.awt.BorderLayout").CENTER)
            caption = self._label(f"{page * 4 + index + 1}. {image['label'][:100]}", size=12)
            caption.setToolTipText(image["label"])
            caption.setBorder(j("javax.swing.BorderFactory").createEmptyBorder(6, 8, 6, 8))
            panel.add(caption, j("java.awt.BorderLayout").SOUTH)
            self.image_grid.add(panel)
        self.image_grid.revalidate()
        self.image_grid.repaint()
