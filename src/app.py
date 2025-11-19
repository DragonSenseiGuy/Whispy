import sys, time, queue, threading
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import webrtcvad
import pyaudio
import collections
import wavio
from src.transcribe import Transcriber
from pynput import keyboard

# --- VAD & Audio Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
SILENCE_THRESHOLD_S = 1.5
VAD_AGGRESSIVENESS = 2
HOTKEY = '<cmd>+<shift>+r'

# --- VAD Audio Thread ---
class VADAudio(QtCore.QThread):
    speech_detected = QtCore.Signal()
    silence_detected = QtCore.Signal()
    audio_data = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)
        self.running = True
        self.speech_has_started = False

    def run(self):
        silent_frames = 0
        num_frames_in_silence_threshold = int(SILENCE_THRESHOLD_S * 1000 / FRAME_DURATION_MS)
        while self.running:
            try:
                data = self.stream.read(FRAME_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_data.emit(audio_chunk)
                is_speech = self.vad.is_speech(data, RATE)
                if is_speech:
                    if not self.speech_has_started:
                        self.speech_detected.emit()
                        self.speech_has_started = True
                    silent_frames = 0
                elif self.speech_has_started:
                    silent_frames += 1
                    if silent_frames > num_frames_in_silence_threshold:
                        self.silence_detected.emit()
                        self.speech_has_started = False
            except IOError:
                pass # Ignore overflows

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.wait()

# --- Recording UI Widget ---
class RecordingUI(QtWidgets.QWidget):
    transcription_requested = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.w, self.h = 200, 75
        self.pill_x0, self.pill_y0, self.pill_x1, self.pill_y1 = 36, 12, self.w - 36, self.h - 12
        self.pill_radius = (self.pill_y1 - self.pill_y0) / 2
        self.bar_count = 31
        self.bar_x, self.base_w = [], []
        margin = 30
        slot_w = (self.pill_x1 - self.pill_x0 - margin * 2) / self.bar_count
        start_x = self.pill_x0 + margin + slot_w * 0.5
        for i in range(self.bar_count):
            self.bar_x.append(start_x + i * slot_w)
            self.base_w.append(max(2.0, slot_w * 0.28))
        self.wave_top = self.pill_y0 + 16
        self.wave_bottom = self.pill_y1 - 16
        self.wave_mid = (self.wave_top + self.wave_bottom) / 2
        self.max_energy, self.noise_floor = 1e-8, 1e-4
        self.heights = np.zeros(self.bar_count, dtype=float)
        self.targets = np.zeros(self.bar_count, dtype=float)
        self.scroll_offset, self.scroll_speed, self.sensitivity = 0.0, 2.4, 2.2
        self.pill_bg, self.border, self.active = QtGui.QColor("#fbf7e4"), QtGui.QColor("#0f0f0d"), QtGui.QColor("#e53935")
        
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.setGeometry((screen.width() - self.w) // 2, screen.height() - self.h - 40, self.w, self.h)
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.q = collections.deque(maxlen=5)
        self.audio_frames = []
        self.is_recording = False
        self.last_time = time.time()

    def start_recording(self):
        self.audio_frames = []
        self.is_recording = True
        self.last_time = time.time()
        self.timer.start(16)
        self.show()
        self.raise_()
        self.activateWindow()

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        self.timer.stop()
        self.hide()
        if self.audio_frames:
            audio_data = np.concatenate(self.audio_frames)
            self.transcription_requested.emit(audio_data)

    def process_audio_data(self, data):
        if self.is_recording: self.audio_frames.append(data)
        if data.size == 0: return
        data_float = data.astype(np.float32)
        rms = float(np.sqrt((data_float * data_float).mean()))
        self.noise_floor = self.noise_floor * 0.995 + rms * 0.005
        active = rms > max(self.noise_floor * 1.8, 1e-6)
        w = data * np.hanning(len(data))
        fft = np.abs(np.fft.rfft(w))
        m = np.max(fft) if fft.size else 0.0
        if m > self.max_energy: self.max_energy = m
        self.max_energy *= 0.995
        bins = len(fft)
        if bins <= 0: return
        idxs = np.linspace(0, bins - 1, self.bar_count)
        interp = np.interp(idxs, np.arange(bins), fft)
        if not active: interp *= 0.0
        interp /= (self.max_energy + 1e-12)
        self.q.append(np.clip(interp, 0.0, 1.0))

    def _gk(self, n=9, s=1.6):
        half = n // 2
        xs = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-(xs**2) / (2 * s * s))
        return k / k.sum()

    def animate(self):
        now = time.time()
        dt = max(1e-6, now - self.last_time)
        self.last_time = now
        if self.q:
            new = self.q.popleft()
            mx = new.max() + 1e-12
            new /= mx
            spread = np.convolve(new, self._gk(), mode="same") * self.sensitivity
            self.targets = self.targets * 0.36 + np.clip(spread, 0.0, 1.0) * 0.64
        decay = 0.72
        self.scroll_offset = (self.scroll_offset + self.scroll_speed * dt * 60.0) % self.bar_count
        for i in range(self.bar_count):
            sample_idx = (i - self.scroll_offset) % self.bar_count
            i0, i1 = int(sample_idx) % self.bar_count, (int(sample_idx) + 1) % self.bar_count
            f = sample_idx - int(sample_idx)
            tgt = (1 - f) * self.targets[i0] + f * self.targets[i1]
            self.heights[i] = self.heights[i] * decay + tgt * (1 - decay)
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.pill_x0, self.pill_y0, self.pill_x1 - self.pill_x0, self.pill_y1 - self.pill_y0)
        pen = QtGui.QPen(self.border, 3)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(self.pill_bg)
        p.drawRoundedRect(r, self.pill_radius, self.pill_radius)
        min_len, max_len = (self.wave_bottom - self.wave_top) * 0.06, (self.wave_bottom - self.wave_top) * 0.9
        for idx, x in enumerate(self.bar_x):
            hn = float(self.heights[idx])
            length = min_len + (max_len - min_len) * max(0.0, min(1.0, hn))
            y0, y1 = self.wave_mid - length / 2, self.wave_mid + length / 2
            stroke = max(1.0, self.base_w[idx] * (1.0 + 1.2 * hn))
            t = min(1.0, hn * 0.95)
            c1, c2 = self.border, self.active
            r, g, b = [int(c1.red() + (c2.red() - c1.red()) * t), int(c1.green() + (c2.green() - c1.green()) * t), int(c1.blue() + (c2.blue() - c1.blue()) * t)]
            pen = QtGui.QPen(QtGui.QColor(r, g, b), stroke)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x, y0), QtCore.QPointF(x, y1))

# --- Hotkey Listener ---
class HotkeyListener(QtCore.QObject):
    hotkey_pressed = QtCore.Signal()
    def __init__(self, hotkey_str):
        super().__init__()
        self.hotkey_str = hotkey_str
        self.listener = None

    @QtCore.Slot()
    def start(self):
        self.listener = keyboard.GlobalHotKeys({
            self.hotkey_str: self.on_activate
        })
        self.listener.start()

    def on_activate(self):
        self.hotkey_pressed.emit()

    def stop(self):
        if self.listener: self.listener.stop()

# --- Main Application Controller ---
class AppController(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.state = "IDLE"
        self.ui = RecordingUI()
        self.vad_thread = VADAudio()
        self.transcriber = Transcriber()

        self.vad_thread.silence_detected.connect(self.on_silence_detected)
        self.vad_thread.audio_data.connect(self.ui.process_audio_data)
        self.ui.transcription_requested.connect(self.handle_transcription)
        self.vad_thread.start()

    @QtCore.Slot()
    def toggle_recording(self):
        if self.state == "IDLE":
            print("Starting recording...")
            self.state = "RECORDING"
            self.ui.start_recording()
        elif self.state == "RECORDING":
            print("Stopping recording...")
            self.state = "IDLE"
            self.ui.stop_recording()

    @QtCore.Slot()
    def on_silence_detected(self):
        if self.state == "RECORDING":
            print("Silence detected, stopping recording...")
            self.state = "IDLE"
            self.ui.stop_recording()

    @QtCore.Slot(np.ndarray)
    def handle_transcription(self, audio_data):
        print("Transcription requested...")
        thread = threading.Thread(target=self._transcribe_task, args=(audio_data,))
        thread.daemon = True
        thread.start()

    def _transcribe_task(self, audio_data):
        wavio.write("temp_audio.wav", audio_data, RATE, sampwidth=2)
        text, _ = self.transcriber.transcribe_audio("temp_audio.wav")
        if text.strip():
            print(f"Transcription: {text}")
        else:
            print("Transcription was empty.")

    def stop(self):
        self.vad_thread.stop()

def create_app_icon():
    pixmap = QtGui.QPixmap(64, 64) # Larger size for app icon
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setBrush(QtGui.QColor("white"))
    painter.drawEllipse(4, 4, 56, 56) # Adjusted for 64x64
    painter.setBrush(QtGui.QColor("#e53935"))
    painter.drawEllipse(12, 12, 40, 40) # Adjusted for 64x64
    painter.end()
    return QtGui.QIcon(pixmap)

def create_tray_icon(app, icon):
    tray_icon = QtWidgets.QSystemTrayIcon(icon, app)
    tray_icon.setToolTip(f"Whispy Recorder (Hotkey: {HOTKEY})")
    menu = QtWidgets.QMenu()
    quit_action = menu.addAction("Quit")
    quit_action.triggered.connect(app.quit)
    tray_icon.setContextMenu(menu)
    return tray_icon

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Whispy")
    app.setQuitOnLastWindowClosed(False)

    app_icon = create_app_icon()
    app.setWindowIcon(app_icon)

    controller = AppController()

    hotkey_thread = QtCore.QThread()
    hotkey_listener = HotkeyListener(HOTKEY)
    hotkey_listener.moveToThread(hotkey_thread)

    hotkey_listener.hotkey_pressed.connect(controller.toggle_recording)
    hotkey_thread.started.connect(hotkey_listener.start)
    
    app.aboutToQuit.connect(hotkey_listener.stop)
    app.aboutToQuit.connect(hotkey_thread.quit)
    app.aboutToQuit.connect(hotkey_thread.wait)
    app.aboutToQuit.connect(controller.stop)

    hotkey_thread.start()

    tray_icon = create_tray_icon(app, app_icon)
    tray_icon.show()
    
    print(f"Whispy is running in the background. Press {HOTKEY} to start/stop recording.")
    print("Right-click the tray icon to quit.")

    sys.exit(app.exec())
