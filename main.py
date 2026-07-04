import sys
import signal
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit, QTextEdit, QVBoxLayout
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPalette
from pynput import keyboard
from openai import OpenAI

# Change this to whatever model is currently loaded/served in LM Studio's
# Developer tab. Run `curl http://localhost:1234/v1/models` to see the
# exact ID LM Studio expects — it must match exactly, no guessing.
MODEL_NAME = "qwen3-coder-30b-a3b-instruct"

def get_loaded_model(fallback=MODEL_NAME):
    """Ask LM Studio which model is currently loaded instead of hardcoding
    one. This means switching models in LM Studio's UI — Qwen, Gemma,
    anything else — just works without editing this file at all.
    """
    try:
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return fallback

class PopupToggleSignal(QObject):
    """QObject that emits a signal when the hotkey is pressed.
    This is needed because Qt GUI operations (like showing/hiding widgets)
    must be performed in the main thread. pynput callbacks run in a separate
    thread, so we can't directly call show() or hide() from that context.
    Instead, we use Qt's signal/slot mechanism to safely communicate
    between the hotkey listener thread and the main GUI thread.
    """
    toggle_signal = pyqtSignal()

class StreamUpdateSignal(QObject):
    """QObject that handles streaming updates from LM Studio.
    API calls run in a background thread so the UI never blocks, but
    updating the QTextEdit must happen on the main thread. These signals
    carry data safely across that thread boundary.
    """
    chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()

class PopupWindow(QWidget):
    def __init__(self, toggle_signal):
        super().__init__()
        self.toggle_signal = toggle_signal
        self.stream_signals = StreamUpdateSignal()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Set translucent background for macOS-style corners
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border-radius: 12px;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #404040;
                padding: 10px;
                font-size: 16px;
                border-radius: 8px;
            }
            QLineEdit::placeholder {
                color: #aaaaaa;
                font-style: italic;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #404040;
                padding: 10px;
                font-size: 14px;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Read to ask bro?")
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("output here bro")
        
        layout.addWidget(self.line_edit)
        layout.addWidget(self.text_edit)
        
        self.setLayout(layout)
        self.position_top_right()
        # Fade-in animation
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Signal/slot wiring
        self.toggle_signal.toggle_signal.connect(self.show_popup)
        self.stream_signals.chunk_received.connect(self.append_to_text_edit)
        self.stream_signals.stream_finished.connect(self.stream_complete)
        self.line_edit.returnPressed.connect(self.handle_enter_key)
        self.hide()

    def center_on_screen(self):
        primary_screen = QApplication.primaryScreen().geometry()
        window_width = 400
        window_height = 300
        x = (primary_screen.width() - window_width) // 2
        y = (primary_screen.height() - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)


    def position_top_right(self):
        primary_screen = QApplication.primaryScreen().availableGeometry()
        window_width = 400
        window_height = 300
        x = primary_screen.width() - window_width - 20
        y = 20
        self.setGeometry(x, y, window_width, window_height)


    def _enable_cross_space_overlay(self):
        """Make this a true Spotlight-style overlay: visible across every
        macOS Space and over full-screen apps, not just 'on top' within
        the current desktop. WindowStaysOnTopHint alone only keeps you
        above normal windows on the same Space — it won't follow you
        when you switch Spaces or when another app goes full-screen.
        """
        try:
            import objc
            from AppKit import (
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSPopUpMenuWindowLevel,
            )
            ns_view = objc.objc_object(c_void_p=int(self.winId()))
            ns_window = ns_view.window()
            if ns_window is not None:
                ns_window.setCollectionBehavior_(
                    NSWindowCollectionBehaviorCanJoinAllSpaces |
                    NSWindowCollectionBehaviorFullScreenAuxiliary
                )
                ns_window.setLevel_(NSPopUpMenuWindowLevel)
        except Exception as e:
            # Falls back to plain WindowStaysOnTopHint behavior if pyobjc
            # isn't installed, or if this fails for any other reason.
            print(f"[Overlay] Native macOS overlay behavior not applied: {e}")

    def show_popup(self):
        """Slot — runs on the main thread when the hotkey fires."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.animation.start()

    def append_to_text_edit(self, text):
        """Slot — runs on the main thread for every streamed chunk."""
        self.text_edit.insertPlainText(text)

    def stream_complete(self):
        self.text_edit.append("\n--- Stream completed ---")

    def handle_enter_key(self):
        prompt = self.line_edit.text().strip()
        if not prompt:
            return
        self.text_edit.clear()
        self.text_edit.append(f"Prompt: {prompt}\n")
        stream_thread = threading.Thread(target=self.stream_prompt, args=(prompt,))
        stream_thread.daemon = True
        stream_thread.start()

    def stream_prompt(self, prompt):
        """Runs on a background thread — never touch widgets directly here."""
        try:
            client = OpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio"
            )
            stream = client.chat.completions.create(
                model=get_loaded_model(),  # auto-detects whatever's loaded in LM Studio
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    self.stream_signals.chunk_received.emit(content)
            # FIX: this was never called, so "Stream completed" never showed
            self.stream_signals.stream_finished.emit()
        except Exception as e:
            # FIX: there was no except block at all — a dead server used to
            # hang the thread silently with no feedback in the UI
            error_msg = (
                f"\n[Error: could not reach LM Studio — {e}]\n"
                f"Is the server running on http://localhost:1234?"
            )
            self.stream_signals.chunk_received.emit(error_msg)
            self.stream_signals.stream_finished.emit()

    def keyPressEvent(self, event):
        """FIX: this was empty (just glitched-out repeated comments)."""
        if event.key() == Qt.Key.Key_Escape:
            self.line_edit.clear()
            self.text_edit.clear()
            self.hide()
        else:
            super().keyPressEvent(event)

    def start_hotkey_listener(self, toggle_signal_obj):
        """FIX: this whole function was missing — no hotkey detection existed."""
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
    toggle_signal_obj = PopupToggleSignal()
    popup = PopupWindow(toggle_signal_obj)
    hotkey_listener = popup.start_hotkey_listener(toggle_signal_obj)
    
    # FIX: signal_handler was a broken, unwired method on the wrong class.
    # Ctrl+C in the terminal needs a real SIGINT handler, and Qt's event loop
    # needs a recurring timer so Python actually gets a chance to run it.
    signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
    keep_alive_timer = QTimer()
    keep_alive_timer.timeout.connect(lambda: None)
    keep_alive_timer.start(200)
    
    sys.exit(app.exec())
