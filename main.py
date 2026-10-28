#!/usr/bin/env python3
"""
macOS Sequoia Liquid Glass Spotlight AI Assistant for LM Studio
---------------------------------------------------------------
Features:
- True native macOS NSVisualEffectView acrylic blur positioned BEHIND Qt widgets
- Directional specular 1px top highlight & native macOS window shadow
- pynput global background listener on Cmd + Shift + K
- Real-time streaming with live tokens/sec speed & token counter
- Slash commands: /clear, /reset, /fix, /system <prompt>, /model
- Quick prompt buttons & 1-click clipboard copy
- Clean direct layout on self so widgets are never covered or clipped
"""

import sys
import time
import signal
import threading
from ctypes import c_void_p

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLineEdit, QTextEdit, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton
)
from PyQt6.QtCore import (
    Qt, QObject, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve
)
from PyQt6.QtGui import QTextCursor
from pynput import keyboard
from openai import OpenAI

# Default fallback model ID (strictly l3-8b-stheno-v3.2)
MODEL_NAME = "l3-8b-stheno-v3.2"
MAX_HISTORY_MESSAGES = 24
LM_STUDIO_URL = "http://localhost:1234/v1"


def get_loaded_model(fallback=MODEL_NAME):
    """
    Asks LM Studio which model is currently loaded and served.
    If 'l3-8b-stheno-v3.2' is running or available, selects it;
    otherwise uses the currently loaded model in LM Studio,
    or falls back to 'l3-8b-stheno-v3.2'.
    """
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        models = client.models.list()
        if models.data:
            # 1. If l3-8b-stheno-v3.2 is among the active models, prioritize it
            for m in models.data:
                if "stheno" in m.id.lower() or "l3-8b-stheno" in m.id.lower():
                    return m.id
            # 2. Otherwise use the currently loaded active model from LM Studio
            return models.data[0].id
    except Exception:
        pass
    return fallback


class PopupToggleSignal(QObject):
    """Thread-safe signal to trigger window visibility from pynput listener."""
    toggle_signal = pyqtSignal()


class StreamUpdateSignal(QObject):
    """Signals for streaming chunks, speed calculation, and completion."""
    chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal(str)
    speed_updated = pyqtSignal(float, int)  # tokens_per_sec, total_tokens


class PopupWindow(QWidget):
    def __init__(self, toggle_signal):
        super().__init__()
        self.setObjectName("PopupWindow")
        self.toggle_signal = toggle_signal
        self.stream_signals = StreamUpdateSignal()

        self.chat_history = []
        self.system_prompt = "You are a fast, concise Spotlight assistant. Provide clear, direct answers."
        self.is_streaming = False
        self.current_response_buffer = ""
        self.current_model = get_loaded_model()

        # Telemetry metrics
        self.stream_start_time = 0.0
        self.token_count = 0

        # Window flags: Frameless, stay-on-top, tool popup (no dock clutter)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._build_ui()
        self.position_center_top()

        # Enable true macOS native acrylic blur behind the widgets
        self._enable_cross_space_overlay()

        # Soft fade-in animation
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(160)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Wire signals
        self.toggle_signal.toggle_signal.connect(self.toggle_popup)
        self.stream_signals.chunk_received.connect(self.append_to_text_edit)
        self.stream_signals.stream_finished.connect(self.stream_complete)
        self.stream_signals.speed_updated.connect(self.update_telemetry)
        self.line_edit.returnPressed.connect(self.handle_enter_key)

        self.hide()

    def _build_ui(self):
        """Constructs layout directly on self so widgets are never obscured."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        # Top Bar: Status + Model + Hotkey
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 0, 2, 0)

        self.status_label = QLabel("⚡ LM Studio")
        self.status_label.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")

        self.model_label = QLabel(f"model: {self.current_model}")
        self.model_label.setStyleSheet("color: rgba(255, 255, 255, 0.45); font-size: 11px;")

        self.hotkey_hint = QLabel("⌘⇧K to toggle · Esc to close")
        self.hotkey_hint.setStyleSheet("color: rgba(255, 255, 255, 0.35); font-size: 10px;")

        top_bar.addWidget(self.status_label)
        top_bar.addWidget(self.model_label)
        top_bar.addStretch()
        top_bar.addWidget(self.hotkey_hint)
        layout.addLayout(top_bar)

        # Main Input LineEdit with specular top highlight
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Ask LM Studio, or type /clear, /fix, /system...")
        layout.addWidget(self.line_edit)

        # Quick Action Buttons Bar
        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(6)

        quick_prompts = [
            ("⚡ /fix", "/fix "),
            ("🔍 Explain", "Explain this concisely: "),
            ("🧼 Refactor", "Refactor and clean this: "),
            ("🗑️ /clear", "/clear")
        ]
        for label, prefix in quick_prompts:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, p=prefix: self._insert_quick_prompt(p))
            actions_bar.addWidget(btn)
        actions_bar.addStretch()
        layout.addLayout(actions_bar)

        # Markdown Output Pane
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Responses appear here in frosted liquid glass...")
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        # Footer: Telemetry + Copy Button
        footer_bar = QHBoxLayout()
        self.telemetry_label = QLabel("Ready")
        self.telemetry_label.setStyleSheet("color: rgba(255, 255, 255, 0.45); font-size: 11px; font-family: monospace;")

        self.copy_btn = QPushButton("Copy Markdown")
        self.copy_btn.clicked.connect(self._copy_output_to_clipboard)

        footer_bar.addWidget(self.telemetry_label)
        footer_bar.addStretch()
        footer_bar.addWidget(self.copy_btn)
        layout.addLayout(footer_bar)

        # Liquid Glass QSS (translucent widgets atop native acrylic blur)
        self.setStyleSheet("""
            QWidget {
                color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            }

            /* Main container: transparent so NSVisualEffectView provides the backdrop blur */
            QWidget#PopupWindow {
                background: transparent;
            }

            /* Search / Input Box with specular top highlight */
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.08);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-top: 1px solid rgba(255, 255, 255, 0.35); /* Specular top light */
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 14.5px;
                selection-background-color: rgba(129, 140, 248, 0.5);
            }
            QLineEdit:focus {
                border: 1px solid rgba(129, 140, 248, 0.85);
                background-color: rgba(255, 255, 255, 0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.40);
            }

            /* Quick Action & Copy Buttons */
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                color: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.16);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.25);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.05);
            }

            /* Frosted Text Output View */
            QTextEdit {
                background-color: rgba(10, 12, 18, 0.42);
                color: #f1f5f9;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 12px;
                padding: 12px 14px;
                font-size: 13.5px;
                line-height: 1.55;
            }
        """)

    def _enable_cross_space_overlay(self):
        """
        Visible across every macOS Space and over full-screen apps,
        with true native NSVisualEffectView acrylic blur placed behind the Qt widgets.
        """
        try:
            import objc
            from AppKit import (
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSPopUpMenuWindowLevel,
                NSVisualEffectView,
                NSVisualEffectMaterialHUDWindow,
                NSVisualEffectBlendingModeBehindWindow,
                NSVisualEffectStateActive,
                NSColor,
                NSViewWidthSizable,
                NSViewHeightSizable,
            )

            # Retrieve underlying native Cocoa NSView and NSWindow
            ns_view = objc.objc_object(c_void_p=int(self.winId()))
            ns_window = ns_view.window()

            if ns_window is not None:
                # 1. Overlay across all macOS Spaces & full-screen apps
                ns_window.setCollectionBehavior_(
                    NSWindowCollectionBehaviorCanJoinAllSpaces |
                    NSWindowCollectionBehaviorFullScreenAuxiliary
                )
                ns_window.setLevel_(NSPopUpMenuWindowLevel)

                # 2. Transparent Cocoa window so NSVisualEffectView provides the background
                ns_window.setOpaque_(False)
                ns_window.setBackgroundColor_(NSColor.clearColor())
                ns_window.setHasShadow_(True)

                # 3. Setup NSVisualEffectView behind the Qt content view
                content_view = ns_window.contentView()
                if content_view is not None:
                    bounds = content_view.bounds()

                    visual_effect_view = NSVisualEffectView.alloc().initWithFrame_(bounds)
                    visual_effect_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
                    visual_effect_view.setMaterial_(NSVisualEffectMaterialHUDWindow)
                    visual_effect_view.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
                    visual_effect_view.setState_(NSVisualEffectStateActive)

                    # Smooth rounded corners matching Qt 20px radius
                    visual_effect_view.setWantsLayer_(True)
                    visual_effect_view.layer().setCornerRadius_(20.0)
                    visual_effect_view.layer().setMasksToBounds_(True)

                    # KEY ARCHITECTURAL FIX:
                    # In Cocoa, superview of contentView is the window frame.
                    # Placing visual_effect_view in superview BELOW content_view
                    # or making visual_effect_view the contentView and adding content_view inside it
                    # ensures Qt widgets are drawn ON TOP of the blur, not covered by it!
                    superview = content_view.superview()
                    if superview is not None:
                        # Insert below content_view (-1 = NSWindowBelow)
                        superview.addSubview_positioned_relativeTo_(visual_effect_view, -1, content_view)
                    else:
                        ns_window.setContentView_(visual_effect_view)
                        content_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
                        visual_effect_view.addSubview_(content_view)

                    ns_window.invalidateShadow()
                    print("[Overlay] Native macOS NSVisualEffectView acrylic blur active behind Qt widgets.")
        except Exception as e:
            print(f"[Overlay] PyObjC / NSVisualEffectView not applied: {e}")

    def position_center_top(self):
        """Positions Spotlight in upper center with macOS proportions."""
        primary_screen = QApplication.primaryScreen().availableGeometry()
        window_width = 540
        window_height = 380
        x = (primary_screen.width() - window_width) // 2
        y = int(primary_screen.height() * 0.16)
        self.setGeometry(x, y, window_width, window_height)

    def toggle_popup(self):
        if self.isVisible():
            self.hide()
        else:
            self.show_popup()

    def show_popup(self):
        self.show()
        self.raise_()
        self.activateWindow()
        self.animation.start()
        self.line_edit.setFocus()
        self.line_edit.selectAll()

    def _insert_quick_prompt(self, prefix):
        if prefix == "/clear":
            self.handle_clear()
            return
        self.line_edit.setText(prefix)
        self.line_edit.setFocus()
        self.line_edit.setCursorPosition(len(prefix))

    def _copy_output_to_clipboard(self):
        text = self.text_edit.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.copy_btn.setText("✓ Copied!")
            QTimer.singleShot(1500, lambda: self.copy_btn.setText("Copy Markdown"))

    def append_to_text_edit(self, text):
        self.current_response_buffer += text
        self.text_edit.setMarkdown(self.current_response_buffer)

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum()
        )

    def update_telemetry(self, speed, tokens):
        self.telemetry_label.setText(f"⚡ {speed:.1f} tok/s · {tokens} tokens")

    def stream_complete(self, full_response):
        self.is_streaming = False
        if full_response:
            self.chat_history.append({"role": "assistant", "content": full_response})
        if len(self.chat_history) > MAX_HISTORY_MESSAGES:
            self.chat_history = self.chat_history[-MAX_HISTORY_MESSAGES:]

        elapsed = max(0.01, time.time() - self.stream_start_time)
        speed = self.token_count / elapsed
        self.telemetry_label.setText(f"✓ Complete · {speed:.1f} tok/s · {self.token_count} tokens")

    def handle_clear(self):
        self.chat_history.clear()
        self.line_edit.clear()
        self.text_edit.clear()
        self.telemetry_label.setText("History cleared")

    def handle_enter_key(self):
        if self.is_streaming:
            return

        prompt = self.line_edit.text().strip()
        if not prompt:
            return

        # 1. Slash commands: /clear or /reset
        if prompt.lower() in ["/clear", "/reset"]:
            self.handle_clear()
            return

        # 2. Slash command: /system <instructions>
        if prompt.lower().startswith("/system"):
            new_sys = prompt[7:].strip()
            if new_sys:
                self.system_prompt = new_sys
                self.text_edit.setMarkdown(f"**System Prompt Updated:**\n> {new_sys}")
            else:
                self.text_edit.setMarkdown(f"**Current System:**\n> {self.system_prompt}")
            self.line_edit.clear()
            return

        # 3. Slash command: /model
        if prompt.lower().startswith("/model"):
            self.current_model = get_loaded_model()
            self.model_label.setText(f"model: {self.current_model}")
            self.text_edit.setMarkdown(f"**Connected Model:** `{self.current_model}`")
            self.line_edit.clear()
            return

        # Refresh currently running model or fallback to l3-8b-stheno-v3.2
        self.current_model = get_loaded_model()
        self.model_label.setText(f"model: {self.current_model}")

        # Regular prompt execution
        self.line_edit.clear()
        self.current_response_buffer = f"**Prompt:** {prompt}\n\n"
        self.text_edit.setMarkdown(self.current_response_buffer)

        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "user", "content": prompt})

        self.is_streaming = True
        self.token_count = 0
        self.stream_start_time = time.time()
        self.telemetry_label.setText(f"Streaming from {self.current_model}...")

        stream_thread = threading.Thread(
            target=self.stream_prompt,
            args=(messages,),
            daemon=True
        )
        stream_thread.start()

    def stream_prompt(self, messages):
        full_response = ""
        token_count = 0
        start_time = time.time()

        try:
            client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
            stream = client.chat.completions.create(
                model=self.current_model or get_loaded_model(),
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    token_count += 1
                    self.token_count = token_count
                    self.stream_signals.chunk_received.emit(delta)

                    # Periodically calculate tokens per second
                    if token_count % 3 == 0:
                        elapsed = max(0.01, time.time() - start_time)
                        self.stream_signals.speed_updated.emit(token_count / elapsed, token_count)

            self.stream_signals.stream_finished.emit(full_response)
        except Exception as e:
            err_msg = (
                f"\n[Error: could not reach LM Studio at {LM_STUDIO_URL}]\n"
                f"Details: {e}\n"
                "Verify that LM Studio's Local Server is running and serving on port 1234."
            )
            self.stream_signals.chunk_received.emit(err_msg)
            self.stream_signals.stream_finished.emit("")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)

    def start_hotkey_listener(self, toggle_signal_obj):
        """Global keyboard listener: Cmd + Shift + K."""
        pressed_keys = set()
        hotkey = {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('k')}

        def on_press(key):
            pressed_keys.add(key)
            if hotkey.issubset(pressed_keys):
                toggle_signal_obj.toggle_signal.emit()

        def on_release(key):
            pressed_keys.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        return listener


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    toggle_signal_obj = PopupToggleSignal()
    popup = PopupWindow(toggle_signal_obj)
    hotkey_listener = popup.start_hotkey_listener(toggle_signal_obj)

    # Allow graceful Ctrl+C in terminal
    signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
    keep_alive_timer = QTimer()
    keep_alive_timer.timeout.connect(lambda: None)
    keep_alive_timer.start(200)

    print("════════════════════════════════════════════════════════════")
    print(" 🚀 macOS Sequoia Liquid Glass Spotlight Running!")
    print(" ⌨️  Hotkey: Press [Cmd + Shift + K] anywhere to toggle.")
    print(" 🛑 Press Ctrl+C in this terminal to exit.")
    print("════════════════════════════════════════════════════════════")

    sys.exit(app.exec())
