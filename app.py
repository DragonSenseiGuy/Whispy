import sys, time, queue, threading
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import webrtcvad
import pyaudio
import collections
import wavio
from transcribe import Transcriber

# Audio settings for VAD
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
SILENCE_THRESHOLD = 2  # Seconds of silence before stopping
VAD_AGGRESSIVENESS = 3

class VADAudio(QtCore.QThread):
    speech_detected = QtCore.Signal()
    silence_detected = QtCore.Signal()
    audio_data = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=FRAME_SIZE)
        self.running = True
        self.speech_has_started = False

    def run(self):
        silent_frames = 0
        num_frames_in_silence_threshold = int(SILENCE_THRESHOLD * 1000 / FRAME_DURATION_MS)

        while self.running:
            try:
                data = self.stream.read(FRAME_SIZE, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, RATE)
                
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_data.emit(audio_chunk)

                if is_speech:
                    if not self.speech_has_started:
                        self.speech_detected.emit()
                        self.speech_has_started = True
                    silent_frames = 0
                else:
                    if self.speech_has_started:
                        silent_frames += 1
                        if silent_frames > num_frames_in_silence_threshold:
                            self.silence_detected.emit()
                            self.speech_has_started = False # Reset for next time
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    # This should not happen with the queue, but as a safeguard
                    print("Input overflowed. Discarding.")
                else:
                    raise


    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class TranscriptionThread(QtCore.QThread):
    transcription_ready = QtCore.Signal(str)

    def __init__(self, audio_file, transcriber):
        super().__init__()
        self.audio_file = audio_file
        self.transcriber = transcriber

    def run(self):
        text, _ = self.transcriber.transcribe_audio(self.audio_file)
        self.transcription_ready.emit(text)

class RecordingUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.w, self.h = 520, 180 # Increased height for transcription
        self.pill_x0, self.pill_y0 = 36, 12
        self.pill_x1, self.pill_y1 = self.w - 36, self.h - 52
        self.pill_radius = (self.pill_y1 - self.pill_y0) / 2
        self.bar_count = 31
        self.bar_x, self.base_w = [], []
        margin = 30
        slot_w = (self.pill_x1 - self.pill_x0 - margin * 2) / self.bar_count
        start_x = self.pill_x0 + margin + slot_w * 0.5
        for i in range(self.bar_count):
            self.bar_x.append(start_x + i * slot_w); self.base_w.append(max(2.0, slot_w * 0.28))
        self.wave_top = self.pill_y0 + 16; self.wave_bottom = self.pill_y1 - 16; self.wave_mid = (self.wave_top + self.wave_bottom) / 2
        self.max_energy = 1e-8; self.noise_floor = 1e-4
        self.heights = np.zeros(self.bar_count, dtype=float); self.targets = np.zeros(self.bar_count, dtype=float)
        self.scroll_offset = 0.0; self.scroll_speed = 2.4; self.sensitivity = 2.2
        self.bg = QtGui.QColor("#123456"); self.pill_bg = QtGui.QColor("#fbf7e4"); self.border = QtGui.QColor("#0f0f0d"); self.active = QtGui.QColor("#e53935")
        
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint); self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry(); x = (screen.width() - self.w) // 2; y = screen.height() - self.h - 40
        self.setGeometry(x, y, self.w, self.h)
        
        self.last_time = time.time()
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.animate); self.timer.start(16)
        self.q = collections.deque(maxlen=5)

        self.audio_frames = []
        self.is_recording = False
        self.transcriber = Transcriber()
        self.transcription_label = QtWidgets.QLabel("", self)
        self.transcription_label.setGeometry(self.pill_x0, self.pill_y1 + 5, self.pill_x1 - self.pill_x0, 40)
        self.transcription_label.setAlignment(QtCore.Qt.AlignCenter)
        self.transcription_label.setStyleSheet("color: white; font-size: 16px;")
        self.transcription_thread = None


    def start_recording(self):
        self.audio_frames = []
        self.is_recording = True
        self.transcription_label.setText("")
        self.show()

    def stop_recording_and_transcribe(self):
        self.is_recording = False
        self.hide() # Hide immediately when silence is detected

        if not self.audio_frames:
            return

        audio_data = np.concatenate(self.audio_frames)
        self.audio_frames = []

        wavio.write("temp_audio.wav", audio_data, RATE, sampwidth=2)

        self.transcription_thread = TranscriptionThread("temp_audio.wav", self.transcriber)
        self.transcription_thread.transcription_ready.connect(self.update_transcription)
        # self.transcription_thread.finished.connect(self.hide) # Remove this line
        self.transcription_thread.start()


    def update_transcription(self, text):
        self.transcription_label.setText(text)
        print(f"Transcription: {text}")
        QtWidgets.QApplication.instance().quit()


    def process_audio_data(self, data):
        if self.is_recording:
            self.audio_frames.append(data)

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
        interp = interp / (self.max_energy + 1e-12)
        interp = np.clip(interp, 0.0, 1.0)
        self.q.append(interp)

    def _gk(self, n=9, s=1.6):
        half = n // 2; xs = np.arange(-half, half + 1, dtype=float); k = np.exp(-(xs**2) / (2 * s * s)); return k / k.sum()

    def animate(self):
        now = time.time(); dt = max(1e-6, now - self.last_time); self.last_time = now
        new = None
        if self.q:
            new = self.q.popleft()

        if new is not None:
            mx = new.max() + 1e-12; new = new / mx
            spread = np.convolve(new, self._gk(), mode="same")
            spread *= self.sensitivity; spread = np.clip(spread, 0.0, 1.0)
            self.targets = self.targets * 0.36 + spread * 0.64
        decay = 0.72
        self.scroll_offset = (self.scroll_offset + self.scroll_speed * dt * 60.0) % self.bar_count
        for i in range(self.bar_count):
            sample_idx = (i - self.scroll_offset) % self.bar_count
            i0 = int(sample_idx) % self.bar_count; i1 = (i0 + 1) % self.bar_count
            f = sample_idx - int(sample_idx)
            tgt = (1 - f) * self.targets[i0] + f * self.targets[i1]
            self.heights[i] = self.heights[i] * decay + tgt * (1 - decay)
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        # Main pill background
        r = QtCore.QRectF(self.pill_x0, self.pill_y0, self.pill_x1 - self.pill_x0, self.pill_y1 - self.pill_y0)
        p.setBrush(self.pill_bg); pen = QtGui.QPen(self.border); pen.setWidth(3); pen.setJoinStyle(QtCore.Qt.RoundJoin); pen.setCapStyle(QtCore.Qt.RoundCap); p.setPen(pen); p.drawRoundedRect(r, self.pill_radius, self.pill_radius)
        
        # Waveform bars
        min_len = (self.wave_bottom - self.wave_top) * 0.06; max_len = (self.wave_bottom - self.wave_top) * 0.9
        for idx, x in enumerate(self.bar_x):
            hn = float(self.heights[idx])
            length = min_len + (max_len - min_len) * max(0.0, min(1.0, hn))
            y0 = self.wave_mid - length / 2; y1 = self.wave_mid + length / 2
            stroke = max(1.0, self.base_w[idx] * (1.0 + 1.2 * hn))
            t = min(1.0, hn * 0.95)
            c1, c2 = self.border, self.active
            rcol = int(c1.red() + (c2.red() - c1.red()) * t); gcol = int(c1.green() + (c2.green() - c1.green()) * t); bcol = int(c1.blue() + (c2.blue() - c1.blue()) * t)
            col = QtGui.QColor(rcol, gcol, bcol)
            pen = QtGui.QPen(col); pen.setWidthF(stroke); pen.setCapStyle(QtCore.Qt.RoundCap); p.setPen(pen); p.drawLine(QtCore.QPointF(x, y0), QtCore.QPointF(x, y1))

    def stop(self):
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.wait()

    def closeEvent(self, e):
        self.stop()
        e.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    ui = RecordingUI()
    
    vad_thread = VADAudio()
    vad_thread.speech_detected.connect(ui.start_recording)
    vad_thread.silence_detected.connect(ui.stop_recording_and_transcribe)
    vad_thread.audio_data.connect(ui.process_audio_data)
    
    # Initially hide the UI
    ui.hide()
    
    vad_thread.start()
    
    app.exec()
    
    ui.stop()
    vad_thread.stop()
    vad_thread.wait()
    sys.exit()