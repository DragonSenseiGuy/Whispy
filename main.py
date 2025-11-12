import time
import pyaudio
import numpy as np
import wavio as wv
from transcribe import Transcriber
import collections
import threading
import queue

class Recorder:
    def __init__(self, channels=1, rate=16000, chunk=1024):
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.chunk)
        self.frames_queue = queue.Queue()
        self.recording = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.daemon = True
        self.thread.start()

    def _record_loop(self):
        while self.recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames_queue.put(data)
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    # This should not happen with the queue, but as a safeguard
                    print("Input overflowed. Discarding.")
                else:
                    raise

    def get_frame(self):
        return self.frames_queue.get()

    def stop(self):
        self.recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class App:
    def __init__(self):
        self.recorder = Recorder()
        self.transcriber = Transcriber()
        self.frames = collections.deque(maxlen=int(self.recorder.rate / self.recorder.chunk * 5)) # 5 seconds of audio buffer

    def run(self):
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                data = self.recorder.get_frame()
                self.frames.append(data)
                
                audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                energy = np.abs(audio_data).mean()

                if energy > 500: # Adjust this threshold based on your microphone
                    wv.write("temp_audio.wav", np.frombuffer(b''.join(self.frames), dtype=np.int16), self.recorder.rate, sampwidth=2)
                    
                    text, _ = self.transcriber.transcribe_audio("temp_audio.wav")
                    if text.strip():
                        print(f"Transcription: {text}")
                        self.frames.clear()

        except KeyboardInterrupt:
            print("\nStopping...")
            self.recorder.stop()

if __name__ == "__main__":
    app = App()
    app.run()