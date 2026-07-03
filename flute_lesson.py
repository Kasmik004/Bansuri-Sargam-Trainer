"""
Bansuri Sargam Trainer — PyQt6 GUI
Reuses the DSP pipeline from test3.py (autocorrelation pitch detection,
Butterworth bandpass, energy gating) and walks the player through
lesson sequences note by note.
"""

import sys
import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft, next_fast_len
import sounddevice as sd
from collections import deque
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QFrame, QGridLayout
)
from PyQt6.QtCore import QTimer, Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QImage, QPixmap


# ── DSP Configuration (mirrors test3.py) ─────────────────────────────
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
SA_NOTE = 'B'          # tonic of your bansuri
MIN_FREQ = 150.0
MAX_FREQ = 1500.0
ENERGY_THRESHOLD = 0.0005
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 2
HOLD_FRAMES = 2      # sustained correct frames before advancing

_NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
SA_SEMITONE = _NOTE_TO_SEMITONE[SA_NOTE]

INTERVAL_TO_SARGAM = {
    0: 'Sa', 1: 'Re(k)', 2: 'Re', 3: 'Ga(k)', 4: 'Ga',
    5: 'Ma', 6: "Ma'", 7: 'Pa', 8: 'Dha(k)', 9: 'Dha',
    10: 'Ni(k)', 11: 'Ni'
}


# ── DSP Functions ────────────────────────────────────────────────────

def setup_filter():
    """Create a Butterworth bandpass and zero-initialized filter state."""
    nyq = 0.5 * SAMPLE_RATE
    sos = signal.butter(4, [MIN_FREQ / nyq, MAX_FREQ / nyq],
                        btype='band', output='sos')
    zi = signal.sosfilt_zi(sos) * 0.0
    return sos, zi


def frame_energy(x):
    return float(np.mean(x.astype(np.float64) ** 2))


def parabolic_interpolation(array, x):
    if x < 1 or x > len(array) - 2:
        return x
    alpha, beta, gamma = array[x - 1], array[x], array[x + 1]
    denom = alpha - 2 * beta + gamma
    if denom == 0:
        return x
    return x + 0.5 * (alpha - gamma) / denom


def autocorrelate_normalized(x):
    """FFT-based autocorrelation, O(N log N)."""
    n = len(x)
    x = x.astype(np.float64)
    fft_len = next_fast_len(2 * n - 1)
    X = rfft(x, fft_len)
    corr = irfft(X * np.conj(X), fft_len)[:n]
    return corr / np.arange(n, 0, -1, dtype=np.float64)


def find_fundamental_lag(corr, min_lag, max_lag, threshold_ratio=0.8):
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag:
        return None, 0.0
    search = corr[min_lag:max_lag]
    global_max = np.max(search)
    if global_max <= 0:
        return None, 0.0
    threshold = threshold_ratio * global_max
    for i in range(1, len(search) - 1):
        if (search[i] >= threshold
                and search[i] >= search[i - 1]
                and search[i] >= search[i + 1]):
            return i + min_lag, search[i]
    idx = int(np.argmax(search))
    return idx + min_lag, search[idx]


def pitch_from_autocorrelation(samples):
    corr = autocorrelate_normalized(samples)
    min_lag = int(SAMPLE_RATE / MAX_FREQ)
    max_lag = int(SAMPLE_RATE / MIN_FREQ)
    peak_lag, peak_val = find_fundamental_lag(corr, min_lag, max_lag)
    if peak_lag is None or peak_lag == 0:
        return None, 0.0
    confidence = peak_val / corr[0] if corr[0] != 0 else 0.0
    refined_lag = parabolic_interpolation(corr, peak_lag)
    if refined_lag <= 0:
        return None, 0.0
    return SAMPLE_RATE / refined_lag, confidence


def freq_to_interval(freq):
    """Return the semitone interval (0-11) from Sa."""
    if freq < 20:
        return None
    midi = int(round(12 * np.log2(freq / 440.0) + 69))
    return (midi % 12 - SA_SEMITONE) % 12


def interval_name(interval):
    return INTERVAL_TO_SARGAM.get(interval, '?')


# ── Lesson Definitions ───────────────────────────────────────────────
# Each lesson is a list of intervals (0 = Sa, 2 = Re, … 11 = Ni).

LESSONS = {
    "Aaroh (Ascending)":     [0, 2, 4, 5, 7, 9, 11, 0],
    "Avaroh (Descending)":   [0, 11, 9, 7, 5, 4, 2, 0],
    "Aaroh-Avaroh":          [0, 2, 4, 5, 7, 9, 11, 0,
                              0, 11, 9, 7, 5, 4, 2, 0],
    "Alankar 1 (Step Up)":   [0, 2, 0, 2, 4, 2, 4, 5,
                              4, 5, 7, 5, 7, 9],
    "Alankar 2 (Triplets)":  [0, 2, 4, 2, 4, 5, 4, 5,
                              7, 5, 7, 9, 7, 9, 11],
    "Sa Re Sa":              [0, 2, 0, 2, 0, 2, 0],
    "Pa-Sa Range":           [7, 9, 11, 0, 2, 4, 5, 7],
    "Kal ho na ho":          [0, 11, 0, 11, 0, 11, 0, 4, 2, 0, 11, 9, 11, 9, 11],
    "Krishna":               [7, 11, 0, 11, 0, 11, 7, 11, 0, 11, 0, 11,
                                ]
}

CARDS_PER_ROW = 8


# ── Note Card Widget ─────────────────────────────────────────────────

class NoteCard(QFrame):
    """A styled card showing a single sargam note."""

    UPCOMING = 'upcoming'
    CURRENT = 'current'
    HOLDING = 'holding'
    DONE = 'done'

    _STYLES = {
        UPCOMING: ("background: #2a2a3a; border: 2px solid #444466;"
                   " border-radius: 10px;",
                   "color: #888899;"),
        CURRENT:  ("background: #1a1a4a; border: 3px solid #6688ff;"
                   " border-radius: 10px;",
                   "color: #aaccff;"),
        HOLDING:  ("background: #1a3a1a; border: 3px solid #44cc44;"
                   " border-radius: 10px;",
                   "color: #88ff88;"),
        DONE:     ("background: #1a3a2a; border: 2px solid #338844;"
                   " border-radius: 10px;",
                   "color: #55aa66;"),
    }

    def __init__(self, note_text, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(64, 64))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel(note_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        layout.addWidget(self.label)

        self.set_state(self.UPCOMING)

    def set_state(self, state):
        frame_css, label_css = self._STYLES.get(state, self._STYLES[self.UPCOMING])
        self.setStyleSheet(f"NoteCard {{ {frame_css} }}")
        self.label.setStyleSheet(label_css)


# ── Main Window ──────────────────────────────────────────────────────

class FluteLessonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bansuri Sargam Trainer")
        self.setMinimumSize(640, 480)

        # State
        self.running = False
        self.current_idx = 0
        self.hold_count = 0
        self.lesson_intervals = []
        self.note_cards = []
        self.recent_intervals = deque(maxlen=SMOOTHING_WINDOW)

        # Audio
        self.stream = None
        self.sos, self.zi = setup_filter()
        self.hann_window = np.hanning(BUFFER_SIZE)

        self._build_ui()
        self._load_lesson()

        # Timer fires every ~46 ms (one buffer of audio at 44100 Hz)
        self.timer = QTimer()
        self.timer.setInterval(int(BUFFER_SIZE / SAMPLE_RATE * 1000))
        self.timer.timeout.connect(self._process_audio)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cam_timer = QTimer()
        self.cam_timer.setInterval(33) # ~30 fps
        self.cam_timer.timeout.connect(self._update_camera)
        self.cam_timer.start()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(18)
        root.setContentsMargins(28, 24, 28, 24)

        # Title
        title = QLabel("Bansuri Sargam Trainer")
        title.setFont(QFont('Segoe UI', 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ccddff;")
        root.addWidget(title)

        # Lesson selector row
        sel_row = QHBoxLayout()
        sel_lbl = QLabel("Lesson:")
        sel_lbl.setFont(QFont('Segoe UI', 11))
        sel_lbl.setStyleSheet("color: #aaaacc;")
        self.combo = QComboBox()
        self.combo.setFont(QFont('Segoe UI', 11))
        self.combo.addItems(LESSONS.keys())
        self.combo.currentTextChanged.connect(self._on_lesson_changed)
        self.combo.setStyleSheet(
            "QComboBox { background: #2a2a3a; color: #ccccee;"
            " border: 1px solid #555577; border-radius: 6px;"
            " padding: 5px 10px; min-width: 220px; }"
            " QComboBox::drop-down { border: none; }"
            " QComboBox QAbstractItemView { background: #2a2a3a;"
            " color: #ccccee; selection-background-color: #3a3a6a; }"
        )
        sel_row.addWidget(sel_lbl)
        sel_row.addWidget(self.combo)
        sel_row.addStretch()
        root.addLayout(sel_row)

        # Note cards grid
        self.cards_container = QWidget()
        self.cards_grid = QGridLayout(self.cards_container)
        self.cards_grid.setSpacing(8)
        self.cards_grid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.cards_container)

        # Middle row (Camera + Status)
        middle_row = QHBoxLayout()

        # Camera feed
        self.camera_label = QLabel("Camera not available")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet("background-color: #000; border-radius: 10px; color: #888899;")
        middle_row.addWidget(self.camera_label)

        # Status panel
        status = QFrame()
        status.setStyleSheet(
            "background: #1e1e2e; border-radius: 10px; padding: 12px;"
        )
        sl = QVBoxLayout(status)

        self.target_label = QLabel("Play: --")
        self.target_label.setFont(QFont('Segoe UI', 18, QFont.Weight.Bold))
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_label.setStyleSheet("color: #aaccff;")
        sl.addWidget(self.target_label)

        self.detect_label = QLabel("Waiting...")
        self.detect_label.setFont(QFont('Segoe UI', 13))
        self.detect_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detect_label.setStyleSheet("color: #888899;")
        sl.addWidget(self.detect_label)

        middle_row.addWidget(status)
        root.addLayout(middle_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m notes")
        self.progress.setStyleSheet(
            "QProgressBar { background: #2a2a3a; border: 1px solid #444466;"
            " border-radius: 6px; height: 20px; text-align: center;"
            " color: #ccccee; font-size: 11px; }"
            " QProgressBar::chunk { background: qlineargradient("
            " x1:0,y1:0,x2:1,y2:0, stop:0 #3366cc, stop:1 #44cc88);"
            " border-radius: 5px; }"
        )
        root.addWidget(self.progress)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        self.start_btn.setFixedSize(130, 40)
        self.start_btn.clicked.connect(self._toggle_start)
        self.start_btn.setStyleSheet(
            "QPushButton { background: #3366cc; color: white; border: none;"
            " border-radius: 8px; }"
            " QPushButton:hover { background: #4477dd; }"
        )
        btn_row.addWidget(self.start_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFont(QFont('Segoe UI', 12))
        self.reset_btn.setFixedSize(130, 40)
        self.reset_btn.clicked.connect(self._reset_lesson)
        self.reset_btn.setStyleSheet(
            "QPushButton { background: #444466; color: #ccccee; border: none;"
            " border-radius: 8px; }"
            " QPushButton:hover { background: #555577; }"
        )
        btn_row.addWidget(self.reset_btn)

        root.addLayout(btn_row)

        # Global dark background
        self.setStyleSheet("QMainWindow { background: #14141e; }")
        central.setStyleSheet("background: #14141e;")

    # ── Lesson management ────────────────────────────────────────────

    def _load_lesson(self):
        # Remove old cards
        while self.cards_grid.count():
            item = self.cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        name = self.combo.currentText()
        self.lesson_intervals = LESSONS[name]
        self.note_cards = []

        for i, interval in enumerate(self.lesson_intervals):
            card = NoteCard(interval_name(interval))
            self.note_cards.append(card)
            self.cards_grid.addWidget(card, i // CARDS_PER_ROW,
                                     i % CARDS_PER_ROW)

        self.progress.setMaximum(len(self.lesson_intervals))
        self.current_idx = 0
        self.hold_count = 0
        self.progress.setValue(0)
        self.recent_intervals.clear()
        self._refresh_cards()

    def _on_lesson_changed(self, _text):
        was_running = self.running
        if was_running:
            self._stop_audio()
        self._load_lesson()
        if was_running:
            self._start_audio()

    def _refresh_cards(self):
        for i, card in enumerate(self.note_cards):
            if i < self.current_idx:
                card.set_state(NoteCard.DONE)
            elif i == self.current_idx:
                card.set_state(
                    NoteCard.HOLDING if self.hold_count > 0
                    else NoteCard.CURRENT
                )
            else:
                card.set_state(NoteCard.UPCOMING)

        if self.current_idx < len(self.lesson_intervals):
            t = interval_name(self.lesson_intervals[self.current_idx])
            self.target_label.setText(f"Play:  {t}")
            self.target_label.setStyleSheet("color: #aaccff;")
        else:
            self.target_label.setText("Lesson Complete!")
            self.target_label.setStyleSheet("color: #44cc88;")

    # ── Audio start / stop ───────────────────────────────────────────

    def _toggle_start(self):
        if self.running:
            self._stop_audio()
            self.start_btn.setText("Start")
            self.detect_label.setText("Paused")
        else:
            self._start_audio()
            self.start_btn.setText("Pause")
            self.detect_label.setText("Listening...")

    def _start_audio(self):
        self.sos, self.zi = setup_filter()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=BUFFER_SIZE
        )
        self.stream.start()
        self.running = True
        self.timer.start()

    def _stop_audio(self):
        self.timer.stop()
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _reset_lesson(self):
        self.current_idx = 0
        self.hold_count = 0
        self.progress.setValue(0)
        self.recent_intervals.clear()
        self.detect_label.setText(
            "Listening..." if self.running else "Waiting..."
        )
        self.detect_label.setStyleSheet("color: #888899;")
        self._refresh_cards()

    # ── Camera update ──────────────────────────────────────────────────
    
    def _update_camera(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # OpenCV uses BGR, PyQt needs RGB
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.camera_label.setPixmap(pixmap.scaled(320, 240, Qt.AspectRatioMode.KeepAspectRatio))

    # ── Per-frame audio processing (called by QTimer) ────────────────

    def _process_audio(self):
        if (not self.running or not self.stream
                or self.current_idx >= len(self.lesson_intervals)):
            return

        try:
            data, _ = self.stream.read(BUFFER_SIZE)
        except Exception:
            return

        samples = data[:, 0]
        samples = samples - np.mean(samples)
        samples, self.zi = signal.sosfilt(self.sos, samples, zi=self.zi)

        # Silence gate
        if frame_energy(samples) < ENERGY_THRESHOLD:
            self.detect_label.setText("(silence)")
            self.detect_label.setStyleSheet("color: #666677;")
            self.hold_count = 0
            self.recent_intervals.clear()
            self._refresh_cards()
            return

        windowed = samples * self.hann_window
        freq, confidence = pitch_from_autocorrelation(windowed)

        if freq is None or confidence < CONFIDENCE_THRESHOLD:
            self.detect_label.setText("(breath noise)")
            self.detect_label.setStyleSheet("color: #886644;")
            self.hold_count = 0
            self.recent_intervals.clear()
            self._refresh_cards()
            return

        interval = freq_to_interval(freq)
        if interval is None:
            return

        # Temporal smoothing
        self.recent_intervals.append(interval)
        if len(self.recent_intervals) < SMOOTHING_WINDOW:
            return
        if not all(iv == self.recent_intervals[0]
                   for iv in self.recent_intervals):
            self.detect_label.setText(
                f"Heard: {interval_name(interval)}  (stabilizing...)"
            )
            self.detect_label.setStyleSheet("color: #aaaa44;")
            return

        stable = self.recent_intervals[0]
        detected = interval_name(stable)
        target = self.lesson_intervals[self.current_idx]

        if stable == target:
            self.hold_count += 1
            self.detect_label.setText(
                f"Heard: {detected}  ✓  ({self.hold_count}/{HOLD_FRAMES})"
            )
            self.detect_label.setStyleSheet("color: #44cc44;")
            self._refresh_cards()

            if self.hold_count >= HOLD_FRAMES:
                self.current_idx += 1
                self.hold_count = 0
                self.progress.setValue(self.current_idx)
                self.recent_intervals.clear()
                self._refresh_cards()

                if self.current_idx >= len(self.lesson_intervals):
                    self.detect_label.setText("All notes complete! Great job!")
                    self.detect_label.setStyleSheet("color: #44cc88;")
                    self._stop_audio()
                    self.start_btn.setText("Start")
        else:
            self.hold_count = 0
            need = interval_name(target)
            self.detect_label.setText(
                f"Heard: {detected}   (need {need})"
            )
            self.detect_label.setStyleSheet("color: #cc6644;")
            self._refresh_cards()

    # ── Cleanup ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_audio()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        event.accept()


# ── Entry point ──────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor('#14141e'))
    palette.setColor(QPalette.ColorRole.WindowText, QColor('#ccccee'))
    palette.setColor(QPalette.ColorRole.Base, QColor('#1e1e2e'))
    palette.setColor(QPalette.ColorRole.Text, QColor('#ccccee'))
    app.setPalette(palette)

    window = FluteLessonWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
